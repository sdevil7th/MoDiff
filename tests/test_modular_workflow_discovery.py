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


class ModularWorkflowDiscoveryTests(unittest.TestCase):
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
        self.assertEqual(len(snapshot["contracts"]), 11)
        self.assertEqual(sum(len(item["workflows"]) for item in snapshot["contracts"]), 39)

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
