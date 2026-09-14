import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.modular_workflow_discovery import (
    MODULAR_WORKFLOW_SNAPSHOT,
    ModularWorkflowContractError,
    load_reviewed_modular_workflow_snapshot,
    reviewed_modular_workflow_contract,
    select_modular_workflow,
    validate_modular_workflow_contract,
)


requires_transformers = unittest.skipUnless(
    importlib.util.find_spec("transformers"),
    "requires the staged optional Transformers runtime",
)


class ModularWorkflowDiscoveryTests(unittest.TestCase):
    @requires_transformers
    def test_reviewed_snapshot_matches_the_pinned_upstream_generator(self):
        result = subprocess.run(
            [sys.executable, "scripts/generate_modular_workflow_contracts.py", "--check"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        snapshot = load_reviewed_modular_workflow_snapshot()
        self.assertEqual(snapshot["diffusersRevision"], PINNED_DIFFUSERS_REVISION)
        self.assertEqual(len(snapshot["contracts"]), 34)
        self.assertEqual(sum(len(item["workflows"]) for item in snapshot["contracts"]), 94)

    def test_auto_sequential_loop_state_and_component_reuse_are_normalized(self):
        flux = reviewed_modular_workflow_contract("FluxModularPipeline")
        self.assertEqual(flux["kind"], "auto")
        workflows = {item["taskId"]: item for item in flux["workflows"]}
        self.assertEqual(set(workflows), {"text_to_image", "image_to_image"})
        text = workflows["text_to_image"]
        image = workflows["image_to_image"]
        self.assertEqual(text["executionPipelineClass"], "FluxModularPipeline")
        self.assertIn("prompt", text["requiredInputs"])
        self.assertIn("prompt", text["stateKeys"])
        self.assertIn("image", image["requiredInputs"])
        self.assertTrue(any(step["kind"] == "loop" for step in text["steps"]))
        component_names = {item["name"] for item in flux["components"]}
        self.assertTrue({"transformer", "scheduler", "vae"}.issubset(component_names))
        for component in flux["components"]:
            self.assertEqual(
                component["reuseKey"],
                [component["name"], "loadId", "dtype", "quantization", "device", "offloadMode"],
            )

        wan = reviewed_modular_workflow_contract("WanModularPipeline")
        self.assertEqual(wan["kind"], "sequential")
        self.assertEqual([(item["id"], item["taskId"]) for item in wan["workflows"]], [("default", "text_to_video")])

    def test_schema_round_trip_and_exact_workflow_selection(self):
        contract = reviewed_modular_workflow_contract("FluxModularPipeline")
        round_tripped = json.loads(json.dumps(contract))
        self.assertEqual(validate_modular_workflow_contract(round_tripped), contract)
        selected = select_modular_workflow(
            contract,
            "image_to_image",
            {"prompt": "repaint", "image": object()},
        )
        self.assertEqual(selected["id"], "image2image")
        with self.assertRaisesRegex(ModularWorkflowContractError, "Unknown Modular workflow"):
            select_modular_workflow(contract, "attacker_workflow", {})
        with self.assertRaisesRegex(ModularWorkflowContractError, "missing required inputs: image"):
            select_modular_workflow(contract, "image_to_image", {"prompt": "missing image"})

        minimax = reviewed_modular_workflow_contract("MiniMaxH3ModularPipeline")
        with self.assertRaisesRegex(ModularWorkflowContractError, "requires one complete input set"):
            select_modular_workflow(
                minimax,
                "first_last_frame_to_video_with_audio",
                {"prompt": "missing both keyframes", "num_inference_steps": 50},
            )
        self.assertEqual(
            select_modular_workflow(
                minimax,
                "first_last_frame_to_video_with_audio",
                {"prompt": "first frame", "image": object(), "num_inference_steps": 50},
            )["id"],
            "fl2va",
        )
        self.assertEqual(
            select_modular_workflow(
                minimax,
                "first_last_frame_to_video_with_audio",
                {"prompt": "last frame", "last_image": object(), "num_inference_steps": 50},
            )["id"],
            "fl2va",
        )

    def test_runtime_snapshot_reader_does_not_import_diffusers(self):
        code = (
            "import sys; "
            "from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot; "
            "load_reviewed_modular_workflow_snapshot(); "
            "assert 'diffusers' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_alternative_required_input_sets_fail_closed_when_ambiguous_or_tampered(self):
        contract = json.loads(json.dumps(reviewed_modular_workflow_contract("MiniMaxH3ModularPipeline")))
        workflow = next(item for item in contract["workflows"] if item["id"] == "fl2va")
        workflow["requiredInputAlternatives"] = [
            ["image", "num_inference_steps", "prompt"],
            ["image", "num_inference_steps", "prompt"],
        ]
        with self.assertRaisesRegex(ModularWorkflowContractError, "must be unique"):
            validate_modular_workflow_contract(contract)

    def test_snapshot_reader_rejects_unreviewed_or_malformed_contracts(self):
        snapshot = json.loads(MODULAR_WORKFLOW_SNAPSHOT.read_text(encoding="utf-8"))
        snapshot["contracts"][0]["workflows"][0]["taskId"] = "../../attacker"
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(temporary, "snapshot.json")
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        with self.assertRaises(ModularWorkflowContractError):
            load_reviewed_modular_workflow_snapshot(path)


if __name__ == "__main__":
    unittest.main()
