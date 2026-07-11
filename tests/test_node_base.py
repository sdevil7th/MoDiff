import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.DiffusersAudio.main import Generate as _Generate  # noqa: E402,F401
from modiff.NodeBase import deep_equal  # noqa: E402


class NodeBaseDeepEqualTests(unittest.TestCase):
    def test_nested_audio_arrays_compare_without_image_attributes(self):
        left = {"audio": {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}}
        right = {"audio": {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}}

        self.assertTrue(deep_equal(left, right))

        right["audio"]["samples"][0, 10] = 0.5
        self.assertFalse(deep_equal(left, right))

    def test_direct_node_base_imports_preserve_complete_module_registry(self):
        import_orders = {
            "canonical_first": (
                "from modiff.NodeBase import NodeBase\n"
                "from mellon.NodeBase import NodeBase as LegacyNodeBase"
            ),
            "legacy_first": (
                "from mellon.NodeBase import NodeBase as LegacyNodeBase\n"
                "from modiff.NodeBase import NodeBase"
            ),
        }

        for name, imports in import_orders.items():
            with self.subTest(order=name):
                script = f"""
import json
{imports}
import modules
print(json.dumps({{
    "same_class": NodeBase is LegacyNodeBase,
    "module_count": len(modules.MODULE_MAP),
    "node_count": modules.total_nodes,
}}))
"""
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(
                    payload,
                    {"same_class": True, "module_count": 19, "node_count": 83},
                )


if __name__ == "__main__":
    unittest.main()
