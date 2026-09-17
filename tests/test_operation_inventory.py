"""Every pinned export and declared Auto/Modular task stays visible in coverage."""

import unittest
from pathlib import Path


class OperationInventoryTests(unittest.TestCase):
    def test_inventory_reproduces_from_pinned_source_without_weights(self):
        from modiff.operation_inventory import build_operation_inventory, load_operation_inventory

        generated = build_operation_inventory(Path(__file__).resolve().parents[1])
        self.assertEqual(generated, load_operation_inventory())
        self.assertEqual(len(generated["pipelines"]), 330)
        flux = next(p for p in generated["pipelines"] if p["pipelineClass"] == "FluxPipeline")
        self.assertIn({"task": "text_to_image", "source": "auto", "workflowId": None}, flux["upstreamTasks"])
        qwen = next(p for p in generated["pipelines"] if p["pipelineClass"] == "QwenImageModularPipeline")
        self.assertTrue(any(t["workflowId"] == "inpainting" for t in qwen["upstreamTasks"]))
        self.assertTrue(all(p["upstreamTasks"] for p in generated["pipelines"]))

    def test_conditional_auto_mappings_are_audited_and_new_task_categories_fail_closed(self):
        from modiff.operation_inventory import _auto_tasks

        source = """AUTO_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict([('base', BasePipeline)])
if optional_dependency_available():
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING['conditional'] = ConditionalPipeline
"""
        self.assertEqual(
            _auto_tasks(source), {"BasePipeline": {"text_to_image"}, "ConditionalPipeline": {"text_to_image"}}
        )
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            _auto_tasks(source.replace("TEXT2IMAGE", "UNREVIEWED"))

    def test_inventory_rejects_tampering_even_with_a_recomputed_content_hash(self):
        import json
        import tempfile
        from copy import deepcopy
        from modiff.operation_inventory import _hash, load_operation_inventory

        original = load_operation_inventory()
        for mutate in (
            lambda value: value["pipelines"].append(deepcopy(value["pipelines"][0])),
            lambda value: value["pipelines"][0].update(pipelineClass="constructor"),
            lambda value: value["pipelines"][0].update(coverage="runnable"),
            lambda value: value["pipelines"][0].update(upstreamTasks=[]),
            lambda value: value.update(schemaVersion=True),
            lambda value: value.update(diffusersRevision="a" * 40),
        ):
            value = deepcopy(original)
            mutate(value)
            value["contentHash"] = _hash({key: item for key, item in value.items() if key != "contentHash"})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "inventory.json"
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_operation_inventory(path)
