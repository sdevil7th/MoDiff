"""Read-only setup inspection uses the concrete graph, never a supplier."""
import json
import unittest
from unittest import mock

from modiff.server import WebServer


class Request:
    def __init__(self, value):
        self.body = json.dumps(value).encode()
        self.content_length = len(self.body)

    async def read(self):
        return self.body


class WorkflowSetupTests(unittest.IsolatedAsyncioTestCase):
    async def test_inspection_reuses_runtime_guard_and_groups_exact_shared_requirements(self):
        server = object.__new__(WebServer)
        server.instance = "current-worker"
        requirement = {"schemaVersion": 1, "delivery": "optional_overlay", "requiredNow": True,
                       "profileIds": ["media"], "executionProfileIds": ["media:load"],
                       "state": "staged", "reason": "optional_runtime_staged"}
        nodes = {
            name: {"module": "modules.Video", "action": "Load", "params": {
                "filename": {"value": "@data/input.mp4"},
                "computed": {"sourceId": "first", "sourceKey": "video", "value": "not-authority"},
            } if name == "second" else {"filename": {"value": "@data/input.mp4"}}}
            for name in ("first", "second")
        }
        with mock.patch("modiff.server.loader_optional_runtime_requirement", return_value=requirement) as resolve:
            response = await server.runtime_workflow_setup(Request({"schemaVersion": 1, "graph": {
                "nodes": nodes, "paths": [["first", "second"]],
                "runtimeHints": {"optionalRuntimeRequirement": {"state": "active"}},
            }}))
        self.assertEqual(response.status, 200)
        result = json.loads(response.text)
        self.assertEqual(result["instance"], "current-worker")
        self.assertEqual(result["requirements"], [{"nodeIds": ["first", "second"], "requirement": requirement}])
        self.assertEqual(resolve.call_args_list[1].args, ("modules.Video", "Load", {"filename": "@data/input.mp4"}))
        self.assertRegex(result["graphHash"], r"^sha256:workflow-auto-v1:[a-f0-9]{64}$")

    async def test_invalid_scope_is_rejected_before_observing_packages(self):
        server = object.__new__(WebServer)
        server.instance = "worker"
        invalid = [
            {"schemaVersion": True, "graph": {}},
            {"schemaVersion": 1, "graph": {}, "consent": True},
            {"schemaVersion": 1, "graph": {"nodes": {"draft": {
                "module": "untrusted", "action": "supplier", "params": {}}}, "paths": []}},
            {"schemaVersion": 1, "graph": {"nodes": {"x" * 1025: {
                "module": "untrusted", "action": "supplier", "params": {}}}, "paths": [["x" * 1025]]}},
        ]
        with mock.patch("modiff.server.loader_optional_runtime_requirement") as resolve:
            for body in invalid:
                with self.subTest(body=body):
                    response = await server.runtime_workflow_setup(Request(body))
                    self.assertEqual(response.status, 400)
        resolve.assert_not_called()

    async def test_bounded_json_refuses_duplicate_and_oversized_bodies_before_inspection(self):
        server = object.__new__(WebServer)
        server.instance = "worker"
        duplicate = Request({})
        duplicate.body = b'{"schemaVersion":1,"schemaVersion":1,"graph":{}}'
        duplicate.content_length = len(duplicate.body)
        oversized = Request({})
        oversized.content_length = 8 * 1024 * 1024 + 1025
        with mock.patch("modiff.server.loader_optional_runtime_requirement") as resolve:
            for request in (duplicate, oversized):
                self.assertEqual((await server.runtime_workflow_setup(request)).status, 400)
        resolve.assert_not_called()
