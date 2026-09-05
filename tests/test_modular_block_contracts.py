import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from modiff.modular_block_contracts import (
    MODULAR_BLOCK_CONTRACT_SNAPSHOT,
    ModularBlockContractError,
    load_reviewed_modular_block_snapshot,
    validate_modular_block_snapshot,
)
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION


requires_transformers = unittest.skipUnless(
    importlib.util.find_spec("transformers"),
    "requires the staged optional Transformers runtime",
)


class ModularBlockContractTests(unittest.TestCase):
    def test_snapshot_has_exact_reviewed_coverage_and_deduplicated_definitions(self):
        snapshot = load_reviewed_modular_block_snapshot()
        self.assertEqual(snapshot["schemaVersion"], 1)
        self.assertEqual(snapshot["diffusersRevision"], PINNED_DIFFUSERS_REVISION)
        self.assertEqual(len(snapshot["workflows"]), 94)
        self.assertEqual(len({item["pipelineClass"] for item in snapshot["workflows"]}), 34)
        self.assertEqual(len(snapshot["blockDefinitions"]), 483)
        self.assertEqual(sum(len(item["placements"]) for item in snapshot["workflows"]), 1237)

        definition_ids = {item["id"] for item in snapshot["blockDefinitions"]}
        self.assertTrue(all(item["rootBlockDefinitionId"] in definition_ids for item in snapshot["workflows"]))
        self.assertTrue(
            all(
                placement["blockDefinitionId"] in definition_ids
                for workflow in snapshot["workflows"]
                for placement in workflow["placements"]
            )
        )

    def test_workflow_root_contract_keeps_pipeline_state_initialization_metadata(self):
        snapshot = load_reviewed_modular_block_snapshot()
        definitions = {item["id"]: item for item in snapshot["blockDefinitions"]}
        flux = next(
            item
            for item in snapshot["workflows"]
            if item["pipelineClass"] == "FluxModularPipeline" and item["workflowId"] == "text2image"
        )
        root = definitions[flux["rootBlockDefinitionId"]]

        self.assertEqual(root["className"], "SequentialPipelineBlocks")
        self.assertIn("prompt", {field["name"] for field in root["inputs"]})
        self.assertTrue(any(field["kwargsType"] is not None for field in root["inputs"] + root["outputs"]))

    def test_paths_preserve_exact_sub_block_membership_instead_of_splitting_dotted_names(self):
        snapshot = load_reviewed_modular_block_snapshot()
        flux = next(
            item
            for item in snapshot["workflows"]
            if item["pipelineClass"] == "FluxModularPipeline" and item["workflowId"] == "text2image"
        )
        placements = {item["legacyPath"]: item for item in flux["placements"]}
        self.assertEqual(placements["denoise.input"]["path"], ["denoise.input"])
        self.assertEqual(placements["denoise.denoise"]["path"], ["denoise.denoise"])
        self.assertEqual(
            placements["denoise.denoise.denoiser"]["path"],
            ["denoise.denoise", "denoiser"],
        )

    def test_leaf_contracts_keep_fields_variadic_state_components_and_configs(self):
        snapshot = load_reviewed_modular_block_snapshot()
        definitions = {item["className"]: item for item in snapshot["blockDefinitions"]}
        denoiser = definitions["FluxLoopDenoiser"]
        self.assertIn("latents", {item["name"] for item in denoiser["inputs"]})
        self.assertIn("latents", denoiser["requiredInputs"])
        self.assertIn("transformer", {item["name"] for item in denoiser["components"]})

        variadic = definitions["AnimaDenoiseStep"]
        self.assertEqual(
            [item["kwargsType"] for item in variadic["variadicInputs"]],
            ["denoiser_input_fields"],
        )
        self.assertTrue(any(item["configs"] for item in snapshot["blockDefinitions"]))
        self.assertTrue(
            any(
                component["defaultConfig"] is not None
                for item in snapshot["blockDefinitions"]
                for component in item["components"]
            )
        )

    def test_exact_block_definitions_are_reused_across_overlapping_pipeline_classes(self):
        snapshot = load_reviewed_modular_block_snapshot()
        owners: dict[str, set[str]] = {}
        for workflow in snapshot["workflows"]:
            owners.setdefault(workflow["rootBlockDefinitionId"], set()).add(workflow["pipelineClass"])
            for placement in workflow["placements"]:
                owners.setdefault(placement["blockDefinitionId"], set()).add(workflow["pipelineClass"])
        definitions = {item["id"]: item for item in snapshot["blockDefinitions"]}
        shared_wan = [
            definitions[definition_id]["className"]
            for definition_id, pipeline_classes in owners.items()
            if {
                "WanModularPipeline",
                "WanImage2VideoModularPipeline",
                "Wan22ModularPipeline",
                "Wan22Image2VideoModularPipeline",
            }.issubset(pipeline_classes)
        ]
        self.assertTrue({"WanTextEncoderStep", "WanTextInputStep", "WanVaeDecoderStep"}.issubset(shared_wan))
        self.assertGreaterEqual(sum(len(pipeline_classes) > 1 for pipeline_classes in owners.values()), 124)

    def test_reader_rejects_tampering_and_does_not_import_model_libraries(self):
        snapshot = json.loads(MODULAR_BLOCK_CONTRACT_SNAPSHOT.read_text(encoding="utf-8"))
        snapshot["blockDefinitions"][0]["inputs"].append(
            {
                "name": "attacker",
                "type": "builtins.str",
                "required": False,
                "default": None,
                "description": "",
                "kwargsType": None,
            }
        )
        with self.assertRaisesRegex(ModularBlockContractError, "content hash"):
            validate_modular_block_snapshot(snapshot)

        code = (
            "import sys; "
            "from modiff.modular_block_contracts import load_reviewed_modular_block_snapshot; "
            "load_reviewed_modular_block_snapshot(); "
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

    def test_loader_rejects_unreviewed_workflow_coverage(self):
        snapshot = json.loads(MODULAR_BLOCK_CONTRACT_SNAPSHOT.read_text(encoding="utf-8"))
        snapshot["workflows"].pop()
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(temporary, "blocks.json")
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        with self.assertRaisesRegex(ModularBlockContractError, "exactly cover"):
            load_reviewed_modular_block_snapshot(path)

    @requires_transformers
    def test_snapshot_matches_the_pinned_no_weight_generator(self):
        result = subprocess.run(
            [sys.executable, "scripts/generate_modular_block_contracts.py", "--check"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
