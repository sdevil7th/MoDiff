import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from modiff.template_authoring_specs import (
    TemplateAuthoringSpecError,
    build_template_authoring_spec_ledger,
    render_template_authoring_spec_ledger,
    validate_template_authoring_spec_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "template-authoring-specs.v1.json"
CANDIDATE_PATH = ROOT / "data" / "template-candidate-contracts.v1.json"
MANIFEST_PATH = ROOT / "data" / "workflow-library-manifest.json"


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


class TemplateAuthoringSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = _load(LEDGER_PATH)
        cls.candidates = _load(CANDIDATE_PATH)
        cls.manifest = _load(MANIFEST_PATH)

    def test_checked_in_ledger_is_deterministic_and_valid(self):
        validated = validate_template_authoring_spec_ledger(self.ledger, root=ROOT)
        self.assertEqual(validated, self.ledger)
        self.assertEqual(build_template_authoring_spec_ledger(ROOT), self.ledger)
        self.assertEqual(render_template_authoring_spec_ledger(self.ledger), LEDGER_PATH.read_text())
        self.assertEqual(
            self.ledger["summary"],
            {
                "authoringSpecCount": 135,
                "promptDraftedCount": 120,
                "promptNotApplicableCount": 15,
                "canonicalDefaultsCapturedCount": 135,
                "inputSelectionPendingCount": 83,
                "rightsReviewPendingCount": 135,
                "generationPendingCount": 135,
                "assetCount": 0,
                "authoringStateCounts": {
                    "draft_complete_execution_pending": 52,
                    "draft_complete_input_selection_pending": 83,
                },
            },
        )

    def test_every_hidden_candidate_has_one_original_authoring_spec(self):
        candidate_by_id = {row["canonicalWorkflowId"]: row for row in self.candidates["contracts"]}
        specs_by_id = {row["canonicalWorkflowId"]: row for row in self.ledger["specifications"]}
        self.assertEqual(set(specs_by_id), set(candidate_by_id))
        self.assertEqual(len(specs_by_id), len(self.ledger["specifications"]))
        for workflow_id, specification in specs_by_id.items():
            candidate = candidate_by_id[workflow_id]
            with self.subTest(workflow=workflow_id):
                for field in ("modelType", "mode", "mediaKind"):
                    self.assertEqual(specification[field], candidate[field])
                self.assertEqual(specification["canonicalGraph"]["path"], candidate["graphPath"])
                self.assertEqual(specification["canonicalGraph"]["sha256"], candidate["graphHash"])
                self.assertEqual(
                    specification["defaultsPlan"]["sourceGraphSha256"],
                    candidate["graphHash"],
                )
                self.assertEqual(specification["publicationState"], "hidden_candidate")
                self.assertEqual(specification["assetState"], "not_generated")
                self.assertEqual(specification["assets"], [])

    def test_prompt_drafts_are_original_bounded_and_mode_complete(self):
        no_prompt_modes = {
            "depth_estimation",
            "image_adjustment",
            "image_channels",
            "image_crop",
            "image_filter",
            "image_tile",
            "image_upscale",
            "mask_composite",
            "video_frame_extract",
            "video_stitch",
            "speech_to_text",
            "speech_translation",
            "unconditional_image",
        }
        for specification in self.ledger["specifications"]:
            prompt_plan = specification["promptPlan"]
            with self.subTest(workflow=specification["canonicalWorkflowId"]):
                self.assertEqual(prompt_plan["provenance"], "original_modiff_mode_draft_v1")
                if specification["mode"] in no_prompt_modes:
                    self.assertEqual(prompt_plan["status"], "not_applicable")
                    self.assertIsNone(prompt_plan["prompt"])
                    self.assertFalse(prompt_plan["mustBeLockedInGenerationReceipt"])
                else:
                    self.assertEqual(prompt_plan["status"], "drafted")
                    self.assertIsInstance(prompt_plan["prompt"], str)
                    self.assertGreaterEqual(len(prompt_plan["prompt"]), 40)
                    self.assertLessEqual(len(prompt_plan["prompt"]), 700)
                    self.assertTrue(prompt_plan["mustBeLockedInGenerationReceipt"])
                self.assertNotIn("Comfy", prompt_plan["prompt"] or "")

    def test_input_plans_are_exact_and_keep_selection_and_rights_pending(self):
        candidate_by_id = {row["canonicalWorkflowId"]: row for row in self.candidates["contracts"]}
        media_keys = {
            "requiredImages": "image",
            "requiredVideos": "video",
            "requiredAudio": "audio",
        }
        for specification in self.ledger["specifications"]:
            required_inputs = candidate_by_id[specification["canonicalWorkflowId"]]["requiredInputs"]
            expected = []
            if isinstance(required_inputs, dict):
                minimum_counts = required_inputs.get("minimumCounts", {})
                for key, media_kind in media_keys.items():
                    for field in required_inputs.get(key, []):
                        default_count = (
                            2
                            if specification["mode"] == "multi_image_reference_edit"
                            and field == "referenceImages"
                            else 1
                        )
                        expected.append((field, media_kind, minimum_counts.get(field, default_count)))
            actual = [
                (item["field"], item["mediaKind"], item["minimumCount"])
                for item in specification["inputPlan"]["items"]
            ]
            with self.subTest(workflow=specification["canonicalWorkflowId"]):
                self.assertEqual(actual, expected)
                self.assertEqual(
                    specification["inputPlan"]["status"],
                    "selection_required" if expected else "not_applicable",
                )
                for item in specification["inputPlan"]["items"]:
                    self.assertEqual(item["selectionState"], "pending_original_or_licensed_asset")
                    self.assertEqual(item["rightsState"], "review_required")

    def test_defaults_are_exact_graph_values_not_runtime_or_artifact_switches(self):
        forbidden_parameters = {
            "attention_backend",
            "backend",
            "device",
            "dtype",
            "model_id",
            "offload_mode",
            "pipeline_class",
            "revision",
        }
        captured_count = 0
        for specification in self.ledger["specifications"]:
            fields = specification["defaultsPlan"]["fields"]
            self.assertEqual(
                fields,
                sorted(fields, key=lambda item: (item["role"], item["parameter"])),
            )
            captured_count += len(fields)
            self.assertTrue(forbidden_parameters.isdisjoint(item["parameter"] for item in fields))
        self.assertGreater(captured_count, 900)

    def test_rights_generation_quality_and_publication_claims_remain_closed(self):
        self.assertEqual(
            self.ledger["boundary"],
            {
                "studioVisible": False,
                "publicTemplate": False,
                "selectsInputAssets": False,
                "approvesModelOrMediaRights": False,
                "downloadsModelsOrMedia": False,
                "executesWorkflows": False,
                "generatesAssets": False,
                "approvesQuality": False,
                "importsComfyGraphs": False,
                "copiesComfyPrompts": False,
                "capturesCanonicalDefaults": True,
                "draftsOriginalPromptText": True,
                "maximumClaim": "authoring_spec_drafted",
            },
        )
        for specification in self.ledger["specifications"]:
            claims = specification["claims"]
            self.assertTrue(all(value is False for value in claims.values()))
            self.assertEqual(specification["rightsPlan"]["status"], "review_required")
            self.assertEqual(
                specification["generationPlan"]["status"],
                "blocked_pending_external_evidence",
            )
            self.assertIn(
                "generated_asset_and_human_quality_review_required",
                specification["generationPlan"]["blockers"],
            )

    def test_source_provenance_is_bound_to_exact_checked_in_bytes(self):
        sources = self.ledger["sources"]
        self.assertEqual(
            sources["candidateContracts"]["sha256"],
            "sha256:" + hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            sources["candidateContracts"]["contentHash"],
            self.candidates["contentHash"],
        )
        self.assertEqual(
            sources["workflowManifest"]["sha256"],
            "sha256:" + hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        )

    def test_self_consistent_tampering_fails_against_current_sources(self):
        mutations = []

        prompt = copy.deepcopy(self.ledger)
        prompt["specifications"][0]["promptPlan"]["prompt"] = "Changed prompt"
        mutations.append(prompt)

        rights = copy.deepcopy(self.ledger)
        rights["specifications"][0]["rightsPlan"]["status"] = "approved"
        mutations.append(rights)

        asset = copy.deepcopy(self.ledger)
        asset["specifications"][0]["assetState"] = "generated"
        mutations.append(asset)

        graph = copy.deepcopy(self.ledger)
        graph["specifications"][0]["canonicalGraph"]["sha256"] = "0" * 64
        mutations.append(graph)

        for mutated in mutations:
            _reseal(mutated)
            with self.subTest(mutation=mutated["specifications"][0]["canonicalWorkflowId"]):
                with self.assertRaises(TemplateAuthoringSpecError):
                    validate_template_authoring_spec_ledger(mutated, root=ROOT)

    def test_cli_checks_and_validates_offline(self):
        for option in ("--check", "--validate"):
            result = subprocess.run(
                [sys.executable, "scripts/generate_template_authoring_specs.py", option],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
