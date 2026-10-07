"""Every pinned export and declared Auto/Modular task stays visible in coverage."""

import unittest
from pathlib import Path
from unittest.mock import patch


class OperationInventoryTests(unittest.TestCase):
    def test_inventory_reproduces_from_pinned_source_without_weights(self):
        from modiff.operation_inventory import build_operation_inventory, load_operation_inventory

        generated = build_operation_inventory(Path(__file__).resolve().parents[1])
        self.assertEqual(generated, load_operation_inventory())
        self.assertEqual(len(generated["pipelines"]), 334)
        flux = next(p for p in generated["pipelines"] if p["pipelineClass"] == "FluxPipeline")
        self.assertIn({"task": "text_to_image", "source": "auto", "workflowId": None}, flux["upstreamTasks"])
        qwen = next(p for p in generated["pipelines"] if p["pipelineClass"] == "QwenImageModularPipeline")
        self.assertTrue(any(t["workflowId"] == "inpainting" for t in qwen["upstreamTasks"]))
        self.assertTrue(all(p["upstreamTasks"] for p in generated["pipelines"]))

    def test_inventory_reproduces_with_crlf_upstream_python_source(self):
        from modiff.operation_inventory import build_operation_inventory, load_operation_inventory
        from modiff.upstream_coverage import installed_diffusers_source

        root = Path(__file__).resolve().parents[1]
        source = installed_diffusers_source()
        python_sources = {source / "__init__.py", source / "pipelines" / "auto_pipeline.py"}
        read_bytes = Path.read_bytes

        def windows_source_bytes(path):
            body = read_bytes(path)
            if path in python_sources:
                return body.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            return body

        # Simulate the real pinned wheel's Windows Git checkout without changing
        # installed files, model caches, or the reviewed JSON snapshot.
        with patch.object(Path, "read_bytes", windows_source_bytes):
            generated = build_operation_inventory(root)
        self.assertEqual(generated, load_operation_inventory())

    def test_inventory_source_receipts_still_detect_code_and_snapshot_byte_changes(self):
        from modiff.operation_inventory import build_operation_inventory, load_operation_inventory
        from modiff.upstream_coverage import installed_diffusers_source

        root = Path(__file__).resolve().parents[1]
        source = installed_diffusers_source()
        reviewed = load_operation_inventory()
        read_bytes = Path.read_bytes
        for name, changed_path, extra_bytes in (
            ("exports", source / "__init__.py", b"\n# source drift\n"),
            ("autoTasks", source / "pipelines" / "auto_pipeline.py", b"\n# source drift\n"),
            ("modularTasks", root / "data" / "modular-workflow-contracts.json", b" "),
        ):
            def changed_bytes(path):
                body = read_bytes(path)
                return body + extra_bytes if path == changed_path else body

            with self.subTest(source=name), patch.object(Path, "read_bytes", changed_bytes):
                generated = build_operation_inventory(root)
            self.assertNotEqual(generated["sources"][name], reviewed["sources"][name])
            self.assertNotEqual(generated["contentHash"], reviewed["contentHash"])
            self.assertEqual(generated["pipelines"], reviewed["pipelines"])

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
