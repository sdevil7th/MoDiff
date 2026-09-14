import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

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

    def test_interrupted_loop_resumes_after_its_last_completed_checkpoint(self):
        graph = self._graph()
        prepared = self.server._prepare_graph_loops(graph)
        original_execute = self.server.execute_node
        completed_result_indexes = []

        def interrupt_after_first_result(node_id, node, sid, **kwargs):
            output = original_execute(node_id, node, sid, **kwargs)
            if node_id == "result":
                index = self.server.node_cache["index"].output["index"]
                completed_result_indexes.append(index)
                if index == 0:
                    self.server.interrupt_flag = True
            return output

        self.server.execute_node = interrupt_after_first_result
        with self.assertRaisesRegex(InterruptedError, "interrupted before iteration 2"):
            self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

        self.server.interrupt_flag = False
        self.server.node_cache.clear()
        result = self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

        self.assertEqual(result["collection"], [0, 1, 2])
        self.assertEqual(completed_result_indexes, [0, 1, 2])
        self.assertTrue(any("Resuming after 1" in item.get("message", "") for item in self.messages))

    def test_durable_loop_resumes_retained_segments_after_process_replacement(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            retained_root = root / "retained"
            retained_root.mkdir()
            assets = []
            for index in range(2):
                path = retained_root / f"segment-{index}.mp4"
                path.write_bytes(b"retained-video-fixture")
                assets.append(
                    {
                        "schema_version": 1,
                        "asset_id": f"segment-{index}",
                        "storage": "file",
                        "path": str(path),
                        "media_type": "video",
                        "width": 8,
                        "height": 6,
                        "fps": 8.0,
                        "frame_count": 8,
                        "duration_seconds": 1.0,
                        "task_id": "first-worker",
                        "temporary": True,
                        "pinned": True,
                    }
                )

            graph = {
                "sid": "test",
                "nodes": {
                    "items": {
                        "module": "modules.WorkflowControl",
                        "action": "LoopItems",
                        "params": {"collection": _param(assets), "item_index": _param(0)},
                    },
                    "result": {
                        "module": "modules.WorkflowControl",
                        "action": "LoopResult",
                        "params": {
                            "value_input": _param(source_id="items", source_key="item"),
                            "stop_input": _param(False),
                        },
                    },
                },
                "paths": [["items", "result"]],
                "loops": [
                    {
                        "id": "durable-video-loop",
                        "bodyNodeIds": ["items", "result"],
                        "iterations": 2,
                        "maxIterations": 2,
                        "itemNodeId": "items",
                        "resultNodeId": "result",
                        "iterationMode": "collection",
                        "carry": False,
                        "collect": True,
                        "durable": True,
                    }
                ],
            }
            identity = {"workflowTabId": "long-video", "runInputHash": "exact-input-hash"}
            self.server.data_dir = str(root / "data")
            self.server.current_task.update({"task_id": "first-worker", "runtimeHints": identity})
            prepared = self.server._prepare_graph_loops(graph)
            original_execute = self.server.execute_node

            def interrupt_after_first_result(node_id, node, sid, **kwargs):
                output = original_execute(node_id, node, sid, **kwargs)
                if node_id == "result" and self.server.node_cache["items"].output["index"] == 0:
                    self.server.interrupt_flag = True
                return output

            self.server.execute_node = interrupt_after_first_result
            with patch("modiff.media_assets.asset_root", return_value=retained_root):
                with self.assertRaisesRegex(InterruptedError, "interrupted before iteration 2"):
                    self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

            replacement = WebServer.__new__(WebServer)
            replacement.modules = MODULE_MAP
            replacement.node_cache = {}
            replacement.data_dir = str(root / "data")
            replacement.current_task = {
                "task_id": "replacement-worker",
                "attempt_index": 0,
                "runtimeHints": identity,
            }
            replacement.interrupt_flag = False
            replacement_messages = []
            replacement.queue_message = lambda message, _sid=None: replacement_messages.append(message)
            replacement_prepared = replacement._prepare_graph_loops(graph)
            result_indexes = []
            replacement_execute = replacement.execute_node

            def record_result(node_id, node, sid, **kwargs):
                output = replacement_execute(node_id, node, sid, **kwargs)
                if node_id == "result":
                    result_indexes.append(replacement.node_cache["items"].output["index"])
                return output

            replacement.execute_node = record_result
            with patch("modiff.media_assets.asset_root", return_value=retained_root):
                result = replacement._execute_graph_loop(
                    replacement_prepared["loops"][0], graph["nodes"], graph["sid"]
                )

            self.assertEqual(result["collection"], assets)
            self.assertEqual(result_indexes, [1])
            self.assertTrue(any("Resuming after 1" in item.get("message", "") for item in replacement_messages))
            checkpoint_files = list((root / "data" / "runtime" / "loop-checkpoints").glob("*.json"))
            self.assertEqual(len(checkpoint_files), 1)
            replacement._clear_loop_checkpoints("replacement-worker")
            self.assertFalse(checkpoint_files[0].exists())

    def test_durable_loop_rejects_non_retained_values_and_missing_run_identity(self):
        graph = self._graph()
        graph["loops"][0]["durable"] = True
        prepared = self.server._prepare_graph_loops(graph)
        with self.assertRaisesRegex(ValueError, "needs workflowTabId and runInputHash"):
            self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

        self.server.current_task["runtimeHints"] = {
            "workflowTabId": "long-video",
            "runInputHash": "exact-input-hash",
        }
        with TemporaryDirectory() as temporary:
            self.server.data_dir = temporary
            with self.assertRaisesRegex(ValueError, "retained file-backed video assets"):
                self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

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

    def test_collection_loop_carries_the_previous_video_boundary_into_a_continuation_job(self):
        opening = Image.new("RGB", (8, 6), "black")
        first_last = Image.new("RGB", (8, 6), "red")
        second_last = Image.new("RGB", (8, 6), "blue")
        jobs = [
            {
                "prompt": "Start the continuous move.",
                "mode": "image_to_video",
                "opening_image": opening,
                "uses_previous_last_frame": False,
            },
            {
                "prompt": "Continue the same move.",
                "mode": "image_to_video",
                "opening_image": None,
                "uses_previous_last_frame": True,
            },
        ]
        graph = {
            "sid": "test",
            "nodes": {
                "carry": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopInput",
                    # Connected outputs are deliberately non-null. The opening
                    # image is a harmless one-frame video seed on iteration 1;
                    # continuation extraction begins only on iteration 2.
                    "params": {"initial": _param([opening])},
                },
                "items": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopItems",
                    "params": {"collection": _param(jobs), "item_index": _param(0)},
                },
                "generate": {
                    "module": "modules.DiffusersVideo",
                    "action": "GenerateShotJob",
                    "params": {
                        "pipeline": _param(object()),
                        "job": _param(source_id="items", source_key="item"),
                        "previous_video": _param(source_id="carry", source_key="value"),
                    },
                },
                "result": {
                    "module": "modules.WorkflowControl",
                    "action": "LoopResult",
                    "params": {
                        "value_input": _param(source_id="generate", source_key="video_out"),
                        "stop_input": _param(False),
                    },
                },
            },
            "paths": [["carry", "items", "generate", "result"]],
            "loops": [
                {
                    "id": "video-loop",
                    "bodyNodeIds": ["carry", "items", "generate", "result"],
                    "iterations": 2,
                    "maxIterations": 2,
                    "inputNodeId": "carry",
                    "itemNodeId": "items",
                    "resultNodeId": "result",
                    "iterationMode": "collection",
                    "carry": True,
                    "collect": True,
                }
            ],
        }
        generated = [
            {"video_out": [opening, first_last], "frames_out": 2},
            {"video_out": [first_last, second_last], "frames_out": 2},
        ]
        prepared = self.server._prepare_graph_loops(graph)

        with patch("modules.DiffusersVideo.main.Generate.execute", side_effect=generated) as execute:
            result = self.server._execute_graph_loop(prepared["loops"][0], graph["nodes"], graph["sid"])

        self.assertEqual(result["collection"], [[opening, first_last], [first_last, second_last]])
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(execute.call_args_list[0].kwargs["reference_images"], [opening])
        self.assertEqual(execute.call_args_list[1].kwargs["reference_images"], [first_last])

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
