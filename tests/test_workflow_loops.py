import unittest

from modiff.server import WebServer
from modules import MODULE_MAP


def _param(value=None, *, source_id=None, source_key=None):
    result = {"value": value}
    if source_id:
        result.update({"sourceId": source_id, "sourceKey": source_key})
    return result


class WorkflowLoopTests(unittest.TestCase):
    def setUp(self):
        self.server = WebServer.__new__(WebServer)
        self.server.modules = MODULE_MAP
        self.server.node_cache = {}
        self.server.current_task = {"task_id": "loop-task", "attempt_index": 0}
        self.server.interrupt_flag = False
        self.messages = []
        self.server.queue_message = lambda message, _sid=None: self.messages.append(message)

    def _graph(self):
        return {
            "sid": "test",
            "nodes": {
                "input": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopInput",
                    "params": {"initial": _param("initial")},
                },
                "index": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopIndex",
                    "params": {
                        "index_value": _param(0),
                        "iteration_count": _param(1),
                    },
                },
                "result": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopResult",
                    "params": {
                        "value_input": _param(source_id="index", source_key="index"),
                        "stop_input": _param(False),
                    },
                },
            },
            "paths": [["input", "index", "result"]],
            "loops": [
                {
                    "id": "loop-container",
                    "bodyNodeIds": ["input", "index", "result"],
                    "iterations": 3,
                    "maxIterations": 10,
                    "inputNodeId": "input",
                    "indexNodeId": "index",
                    "resultNodeId": "result",
                    "carry": True,
                    "collect": True,
                }
            ],
        }

    def test_loop_collects_each_iteration_and_keeps_last_value(self):
        graph = self._graph()
        prepared = self.server._prepare_graph_loops(graph)
        result = self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

        self.assertEqual(result["collection"], [0, 1, 2])
        self.assertEqual(self.server.node_cache["result"].output["value"], 2)
        self.assertEqual(self.server.node_cache["result"].output["collection"], [0, 1, 2])
        loop_progress = [item for item in self.messages if item.get("node") == "loop-container"]
        self.assertEqual(loop_progress[-1]["status"], "succeeded")

    def test_loop_retries_a_failed_iteration_without_losing_previous_results(self):
        graph = self._graph()
        graph["loops"][0]["maxRetries"] = 1
        prepared = self.server._prepare_graph_loops(graph)
        original_execute = self.server.execute_node
        failures = 0

        def flaky_execute(node_id, *args, **kwargs):
            nonlocal failures
            if node_id == "result" and failures == 0:
                failures += 1
                raise RuntimeError("transient")
            return original_execute(node_id, *args, **kwargs)

        self.server.execute_node = flaky_execute
        result = self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

        self.assertEqual(result["collection"], [0, 1, 2])
        self.assertTrue(any("Retrying iteration" in item.get("message", "") for item in self.messages))

    def test_loop_resumes_completed_iterations_after_graph_cache_clear(self):
        graph = self._graph()
        prepared = self.server._prepare_graph_loops(graph)
        original_execute = self.server.execute_node
        failed = False

        def interrupted_attempt(node_id, node, sid, **kwargs):
            nonlocal failed
            overrides = kwargs.get("param_overrides") or {}
            if node_id == "index" and overrides.get("index_value") == 1 and not failed:
                failed = True
                raise RuntimeError("graph retry")
            return original_execute(node_id, node, sid, **kwargs)

        self.server.execute_node = interrupted_attempt
        with self.assertRaisesRegex(RuntimeError, "graph retry"):
            self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

        self.server.node_cache.clear()
        result = self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

        self.assertEqual(result["collection"], [0, 1, 2])
        self.assertTrue(any("Resuming after 1" in item.get("message", "") for item in self.messages))

    def test_loop_rejects_unbounded_or_overlapping_body_contracts(self):
        graph = self._graph()
        graph["loops"][0]["iterations"] = 11
        with self.assertRaisesRegex(ValueError, "between 1"):
            self.server._prepare_graph_loops(graph)

        graph = self._graph()
        graph["loops"].append({**graph["loops"][0], "id": "other-loop"})
        with self.assertRaisesRegex(ValueError, "overlaps another loop"):
            self.server._prepare_graph_loops(graph)

    def test_collection_mode_maps_over_loop_items(self):
        graph = self._graph()
        graph["nodes"]["items"] = {
            "module": "modules.WorkflowControl",
            "action": "LoopItems",
            "params": {
                "collection": _param(["first", "second", "third"]),
                "item_index": _param(0),
            },
        }
        graph["nodes"]["result"]["params"]["value_input"] = _param(source_id="items", source_key="item")
        graph["paths"] = [["input", "index", "items", "result"]]
        loop = graph["loops"][0]
        loop.update(
            {
                "bodyNodeIds": ["input", "index", "items", "result"],
                "iterationMode": "collection",
                "itemNodeId": "items",
            }
        )

        prepared = self.server._prepare_graph_loops(graph)
        result = self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

        self.assertEqual(result["collection"], ["first", "second", "third"])

    def test_only_loop_result_may_cross_the_container_boundary(self):
        graph = self._graph()
        graph["nodes"]["outside"] = {
            "module": "modules.WorkflowControl",
            "action": "LoopResult",
            "params": {
                "value_input": _param(source_id="index", source_key="index"),
                "stop_input": _param(False),
            },
        }
        graph["paths"] = [["input", "index", "result", "outside"]]
        with self.assertRaisesRegex(ValueError, "only expose values through its Loop Result"):
            self.server._prepare_graph_loops(graph)

    def test_strictly_nested_loop_runs_once_per_parent_iteration(self):
        graph = {
            "sid": "test",
            "nodes": {
                "outer-index": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopIndex",
                    "params": {"index_value": _param(0), "iteration_count": _param(1)},
                },
                "inner-index": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopIndex",
                    "params": {"index_value": _param(0), "iteration_count": _param(1)},
                },
                "inner-result": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopResult",
                    "params": {
                        "value_input": _param(source_id="inner-index", source_key="index"),
                        "stop_input": _param(False),
                    },
                },
                "outer-result": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopResult",
                    "params": {
                        "value_input": _param(source_id="inner-result", source_key="value"),
                        "stop_input": _param(False),
                    },
                },
            },
            "paths": [["outer-index", "inner-index", "inner-result", "outer-result"]],
            "loops": [
                {
                    "id": "outer",
                    "bodyNodeIds": ["outer-index", "inner-index", "inner-result", "outer-result"],
                    "iterations": 2,
                    "maxIterations": 10,
                    "indexNodeId": "outer-index",
                    "resultNodeId": "outer-result",
                },
                {
                    "id": "inner",
                    "bodyNodeIds": ["inner-index", "inner-result"],
                    "iterations": 2,
                    "maxIterations": 10,
                    "indexNodeId": "inner-index",
                    "resultNodeId": "inner-result",
                },
            ],
        }

        prepared = self.server._prepare_graph_loops(graph)
        result = self.server._execute_graph_loop(prepared["loops_by_id"]["outer"], graph["nodes"], graph["sid"])

        self.assertEqual(result["collection"], [1, 1])
        inner_completions = [
            item for item in self.messages if item.get("node") == "inner" and item.get("status") == "succeeded"
        ]
        self.assertEqual(len(inner_completions), 2)


if __name__ == "__main__":
    unittest.main()
