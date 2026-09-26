import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from modiff.modular_conditional_contracts import (
    MODULAR_CONDITIONAL_CONTRACT_SNAPSHOT,
    ModularConditionalContractError,
    load_reviewed_modular_conditional_snapshot,
    select_conditional_branch,
    validate_modular_conditional_snapshot,
)
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION


requires_transformers = unittest.skipUnless(
    importlib.util.find_spec("transformers"),
    "requires the staged optional Transformers runtime",
)


class ModularConditionalContractTests(unittest.TestCase):
    def test_snapshot_preserves_every_unpruned_pipeline_and_conditional(self):
        snapshot = load_reviewed_modular_conditional_snapshot()
        self.assertEqual(snapshot["schemaVersion"], 1)
        self.assertEqual(snapshot["diffusersRevision"], PINNED_DIFFUSERS_REVISION)
        self.assertEqual(len(snapshot["pipelines"]), 34)
        self.assertEqual(len(snapshot["blockDefinitions"]), 632)
        self.assertEqual(sum(len(item["placements"]) for item in snapshot["pipelines"]), 1051)
        self.assertEqual(sum(len(item["conditionals"]) for item in snapshot["pipelines"]), 73)
        self.assertEqual(sum(len(item["workflows"]) for item in snapshot["pipelines"]), 94)

    def test_ltx2_duration_keeps_run_and_skip_alternatives(self):
        snapshot = load_reviewed_modular_conditional_snapshot()
        ltx2 = next(item for item in snapshot["pipelines"] if item["pipelineClass"] == "LTX2ModularPipeline")
        duration = next(item for item in ltx2["conditionals"] if item["legacyPath"] == "duration")

        self.assertEqual(duration["branchNames"], ["duration"])
        self.assertEqual(duration["triggerInputs"], ["num_frames"])
        self.assertEqual(select_conditional_branch(duration, []), "duration")
        self.assertIsNone(select_conditional_branch(duration, ["num_frames"]))

        workflows = {item["id"]: item for item in ltx2["workflows"]}
        text_case = workflows["text2video"]["cases"][0]
        context_case = workflows["in_context"]["cases"][0]
        text_selection = next(item for item in text_case["selections"] if item["legacyPath"] == "duration")
        context_selection = next(item for item in context_case["selections"] if item["legacyPath"] == "duration")
        self.assertEqual(text_selection["selectedBlockName"], "duration")
        self.assertIsNone(context_selection["selectedBlockName"])

    def test_invalid_upstream_input_combination_is_a_reviewed_validation_error(self):
        snapshot = load_reviewed_modular_conditional_snapshot()
        cosmos = next(
            item for item in snapshot["pipelines"] if item["pipelineClass"] == "Cosmos3DistilledModularPipeline"
        )
        conditional = next(
            item
            for item in cosmos["conditionals"]
            if {"image", "video"}.issubset({name for name in item["triggerInputs"] if name is not None})
        )
        with self.assertRaisesRegex(ModularConditionalContractError, "either image or video"):
            select_conditional_branch(conditional, ["image", "video"])

    def test_reader_rejects_truth_table_tampering_and_imports_no_model_libraries(self):
        snapshot = json.loads(MODULAR_CONDITIONAL_CONTRACT_SNAPSHOT.read_text(encoding="utf-8"))
        conditional = next(
            selector for pipeline in snapshot["pipelines"] for selector in pipeline["conditionals"]
        )
        conditional["selectionTable"][0]["selectedBlockName"] = "attacker"
        with self.assertRaisesRegex(ModularConditionalContractError, "branch is unknown"):
            validate_modular_conditional_snapshot(snapshot)

        code = (
            "import sys; "
            "from modiff.modular_conditional_contracts import load_reviewed_modular_conditional_snapshot; "
            "load_reviewed_modular_conditional_snapshot(); "
            "assert 'diffusers' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_loader_rejects_missing_pipeline_coverage(self):
        snapshot = json.loads(MODULAR_CONDITIONAL_CONTRACT_SNAPSHOT.read_text(encoding="utf-8"))
        snapshot["pipelines"].pop()
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(temporary, "conditionals.json")
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        with self.assertRaisesRegex(ModularConditionalContractError, "exactly cover"):
            load_reviewed_modular_conditional_snapshot(path)

    @requires_transformers
    def test_ltx2_manual_and_predicted_duration_rebuild_distinct_upstream_pipelines(self):
        from diffusers import LTX2ModularPipeline

        blocks = LTX2ModularPipeline().blocks
        predicted_duration = blocks.get_execution_blocks(prompt=True)
        manual_frames = blocks.get_execution_blocks(prompt=True, num_frames=True)

        self.assertIn("duration", predicted_duration.sub_blocks)
        self.assertNotIn("duration", manual_frames.sub_blocks)
        predicted_pipeline = predicted_duration.init_pipeline()
        manual_pipeline = manual_frames.init_pipeline()
        self.assertIn("duration_head", predicted_pipeline.pretrained_component_names)
        self.assertNotIn("duration_head", manual_pipeline.pretrained_component_names)
        self.assertIn("num_frames", {field.name for field in manual_frames.inputs})

    @requires_transformers
    def test_snapshot_matches_pinned_no_weight_generator(self):
        result = subprocess.run(
            [sys.executable, "scripts/generate_modular_conditional_contracts.py", "--check"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
