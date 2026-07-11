import json
import unittest

import modules as module_registry
from modiff.server import WebServer
from modules.DiffusersImage import ControlGenerate, Edit, Inpaint, MODULE_MAP


class DiffusersImageRegistryTests(unittest.TestCase):
    def test_inherited_nodes_are_registered_with_their_live_contracts(self):
        expected_inputs = {
            "Edit": {"pipeline", "image"},
            "Inpaint": {"pipeline", "image", "mask_image"},
            "ControlGenerate": {"pipeline", "control_image"},
        }

        for name, required_inputs in expected_inputs.items():
            with self.subTest(node=name):
                entry = MODULE_MAP[name]
                self.assertEqual(entry["type"], "custom")
                self.assertEqual(entry["category"], "Diffusers Image")
                self.assertTrue(entry["resizable"])
                self.assertTrue(required_inputs.issubset(entry["params"]))
                self.assertEqual(entry["params"]["images"]["display"], "output")

    def test_registered_classes_can_be_constructed(self):
        for node_class in (Edit, Inpaint, ControlGenerate):
            with self.subTest(node=node_class.__name__):
                node = node_class("registry-probe")
                self.assertEqual(node.node_id, "registry-probe")
                self.assertTrue(node.resizable)


class FakeRequest:
    match_info = {}


class DiffusersImageNodeRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_nodes_route_exposes_flattened_facade_contracts(self):
        server = WebServer(module_registry.MODULE_MAP)
        response = await server.nodes(FakeRequest())
        nodes = json.loads(response.text)["nodes"]

        for action in ("Edit", "Inpaint", "ControlGenerate"):
            with self.subTest(node=action):
                key = f"modules.DiffusersImage.{action}"
                self.assertIn(key, nodes)
                self.assertTrue(nodes[key]["resizable"])
                self.assertEqual(nodes[key]["params"]["images"]["display"], "output")


if __name__ == "__main__":
    unittest.main()
