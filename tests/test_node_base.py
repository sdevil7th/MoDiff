import importlib.util
import json
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.DiffusersAudio.main import Generate as _Generate  # noqa: E402,F401
from modiff.NodeBase import NodeBase, deep_equal, node_message_context  # noqa: E402


class NodeBaseDeepEqualTests(unittest.TestCase):
    def test_dynamic_node_messages_carry_workflow_ownership_and_target_the_originating_session(self):
        class DynamicNode(NodeBase):
            pass

        module_name = ".".join(DynamicNode.__module__.split(".")[:-1])
        definition = {module_name: {"DynamicNode": {"params": {}}}}
        with patch("modiff.NodeBase._module_map", return_value=definition):
            node = DynamicNode("dynamic-node")
        node._sid = "browser-session"
        messages = []
        current_server = SimpleNamespace(
            _current_dynamic_message_identity_payload=lambda: {
                "task_id": "graph-task",
                "workflow_tab_id": "graph-workflow",
                "workflow_canvas_epoch": 4,
            },
            queue_message=lambda message, sid=None: messages.append((message, sid)),
        )

        with patch("modiff.NodeBase._server", return_value=current_server):
            node.set_field_value({"dtype": "float16"})
            with node_message_context(
                {
                    "workflow_tab_id": "field-workflow",
                    "workflow_canvas_epoch": 9,
                }
            ):
                node.set_field_visibility({"dtype": True})

        graph_message, graph_sid = messages[0]
        self.assertEqual(graph_message["task_id"], "graph-task")
        self.assertEqual(graph_message["workflow_tab_id"], "graph-workflow")
        self.assertEqual(graph_message["workflow_canvas_epoch"], 4)
        self.assertEqual(graph_message["sid"], "browser-session")
        self.assertEqual(graph_sid, "browser-session")

        field_message, field_sid = messages[1]
        self.assertNotIn("task_id", field_message)
        self.assertEqual(field_message["workflow_tab_id"], "field-workflow")
        self.assertEqual(field_message["workflow_canvas_epoch"], 9)
        self.assertEqual(field_sid, "browser-session")

    def test_memory_manager_execution_without_node_identity_accepts_default_arguments(self):
        class BareNode(NodeBase):
            pass

        module_name = ".".join(BareNode.__module__.split(".")[:-1])
        definition = {module_name: {"BareNode": {"params": {}}}}
        with patch("modiff.NodeBase._module_map", return_value=definition):
            node = BareNode()

        self.assertEqual(node.mm_exec(lambda: "ok", "cpu"), "ok")
        self.assertEqual(
            node.mm_exec(
                lambda value, *, suffix: f"{value}{suffix}",
                "cpu",
                args=["run"],
                kwargs={"suffix": "-ok"},
            ),
            "run-ok",
        )

    def test_unchanged_deterministic_node_reuses_its_cached_output(self):
        class CachedNode(NodeBase):
            def __init__(self):
                self.execution_count = 0
                super().__init__("cached-node")

            def execute(self, value):
                self.execution_count += 1
                return {"result": value * 2}

        module_name = ".".join(CachedNode.__module__.split(".")[:-1])
        definition = {
            module_name: {
                "CachedNode": {
                    "params": {
                        "value": {"type": "int", "default": 0},
                        "result": {"type": "int", "display": "output"},
                    }
                }
            }
        }
        with patch("modiff.NodeBase._module_map", return_value=definition):
            node = CachedNode()
            self.assertEqual(node(value=4), {"result": 8})
            self.assertTrue(node._has_changed)
            self.assertEqual(node(value=4), {"result": 8})
            self.assertFalse(node._has_changed)
            self.assertEqual(node.execution_count, 1)
            self.assertEqual(node(value=5), {"result": 10})
            self.assertTrue(node._has_changed)
            self.assertEqual(node.execution_count, 2)

    def test_changed_upstream_node_invalidates_consumer_of_same_mutable_object(self):
        from modiff.server import WebServer

        class ConsumerNode(NodeBase):
            def __init__(self, node_id):
                self.execution_count = 0
                super().__init__(node_id)

            def execute(self, pipeline):
                self.execution_count += 1
                return {"result": pipeline["adapter_scale"]}

        module_name = ".".join(ConsumerNode.__module__.split(".")[:-1])
        definition = {
            module_name: {
                "ConsumerNode": {
                    "params": {
                        "pipeline": {"type": "pipeline", "required": True},
                        "result": {"type": "float", "display": "output"},
                    }
                }
            }
        }
        pipeline = {"adapter_scale": 0.5}
        source = SimpleNamespace(
            _has_changed=True,
            module_name="modules.DiffusersImage",
            class_name="LoadAdapter",
            output={"pipeline": pipeline},
        )
        graph_node = {
            "module": module_name,
            "action": "ConsumerNode",
            "params": {
                "pipeline": {
                    "sourceId": "adapter",
                    "sourceKey": "pipeline",
                }
            },
        }

        with patch("modiff.NodeBase._module_map", return_value=definition):
            consumer = ConsumerNode("generate")
            server = object.__new__(WebServer)
            server.modules = definition
            server.node_cache = {"adapter": source, "generate": consumer}

            server.execute_node("generate", graph_node, "test", quiet=True)
            self.assertEqual(consumer.execution_count, 1)

            pipeline["adapter_scale"] = 0.8
            source._has_changed = True
            server.execute_node("generate", graph_node, "test", quiet=True)
            self.assertEqual(consumer.output, {"result": 0.8})
            self.assertEqual(consumer.execution_count, 2)

            source._has_changed = False
            server.execute_node("generate", graph_node, "test", quiet=True)
            self.assertEqual(consumer.execution_count, 2)

    def test_execute_node_replaces_a_document_local_id_with_the_current_action(self):
        from modiff.server import WebServer

        class PriorAction(NodeBase):
            def execute(self, value):
                return {"result": f"prior:{value}"}

        class CurrentAction(NodeBase):
            def execute(self, value):
                return {"result": f"current:{value}"}

        module_name = ".".join(CurrentAction.__module__.split(".")[:-1])
        definition = {
            module_name: {
                "PriorAction": {
                    "params": {
                        "value": {"type": "string", "default": ""},
                        "result": {"type": "string", "display": "output"},
                    }
                },
                "CurrentAction": {
                    "params": {
                        "value": {"type": "string", "default": ""},
                        "result": {"type": "string", "display": "output"},
                    }
                },
            }
        }
        graph_node = {
            "module": module_name,
            "action": "CurrentAction",
            "params": {"value": {"value": "selected"}},
        }

        with (
            patch("modiff.NodeBase._module_map", return_value=definition),
            patch(
                "modiff.server.import_module",
                return_value=SimpleNamespace(CurrentAction=CurrentAction),
            ),
            patch("modiff.server.assert_optional_runtime_ready"),
        ):
            prior = PriorAction("shared-node-id")
            server = object.__new__(WebServer)
            server.modules = definition
            server.node_cache = {"shared-node-id": prior}

            server.execute_node("shared-node-id", graph_node, "test", quiet=True)

        replacement = server.node_cache["shared-node-id"]
        self.assertIsInstance(replacement, CurrentAction)
        self.assertIsNot(replacement, prior)
        self.assertEqual(replacement.output, {"result": "current:selected"})

    def test_cache_ignored_semantic_change_reuses_resident_output_and_invalidates_consumer(self):
        from modiff.server import WebServer

        class ResidentLoader(NodeBase):
            cache_ignored_params = frozenset({"mode"})

            def __init__(self, node_id):
                self.execution_count = 0
                self.pipeline = {"mode": None}
                super().__init__(node_id)

            def __call__(self, **kwargs):
                result = super().__call__(**kwargs)
                result["pipeline"]["mode"] = kwargs["mode"]
                return result

            def execute(self, mode):
                self.execution_count += 1
                return {"pipeline": self.pipeline}

        class ContractConsumer(NodeBase):
            def __init__(self, node_id):
                self.execution_count = 0
                super().__init__(node_id)

            def execute(self, pipeline):
                self.execution_count += 1
                return {"result": pipeline["mode"]}

        module_name = ".".join(ResidentLoader.__module__.split(".")[:-1])
        definition = {
            module_name: {
                "ResidentLoader": {
                    "params": {
                        "mode": {"type": "string", "default": "generate"},
                        "pipeline": {"type": "pipeline", "display": "output"},
                    }
                },
                "ContractConsumer": {
                    "params": {
                        "pipeline": {"type": "pipeline", "required": True},
                        "result": {"type": "string", "display": "output"},
                    }
                },
            }
        }
        graph_node = {
            "module": module_name,
            "action": "ContractConsumer",
            "params": {
                "pipeline": {
                    "sourceId": "loader",
                    "sourceKey": "pipeline",
                }
            },
        }

        with patch("modiff.NodeBase._module_map", return_value=definition):
            loader = ResidentLoader("loader")
            consumer = ContractConsumer("consumer")
            server = object.__new__(WebServer)
            server.modules = definition
            server.node_cache = {"loader": loader, "consumer": consumer}

            first = loader(mode="generate")
            server.execute_node("consumer", graph_node, "test", quiet=True)
            self.assertTrue(loader._has_changed)
            self.assertEqual(loader.execution_count, 1)
            self.assertEqual(consumer.output, {"result": "generate"})

            second = loader(mode="edit")
            server.execute_node("consumer", graph_node, "test", quiet=True)
            self.assertIs(first["pipeline"], second["pipeline"])
            self.assertTrue(loader._has_changed)
            self.assertEqual(loader.execution_count, 1)
            self.assertTrue(consumer._has_changed)
            self.assertEqual(consumer.execution_count, 2)
            self.assertEqual(consumer.output, {"result": "edit"})

            third = loader(mode="edit")
            server.execute_node("consumer", graph_node, "test", quiet=True)
            self.assertIs(second["pipeline"], third["pipeline"])
            self.assertFalse(loader._has_changed)
            self.assertEqual(loader.execution_count, 1)
            self.assertFalse(consumer._has_changed)
            self.assertEqual(consumer.execution_count, 2)

    def test_typed_numeric_value_matches_string_keyed_option_contract(self):
        class NumericOptionNode(NodeBase):
            def execute(self, sample_rate):
                return {"result": sample_rate}

        module_name = ".".join(NumericOptionNode.__module__.split(".")[:-1])
        definition = {
            module_name: {
                "NumericOptionNode": {
                    "params": {
                        "sample_rate": {
                            "type": "int",
                            "default": 48000,
                            "options": {
                                "44100": "44.1 kHz",
                                "48000": "48 kHz",
                            },
                        },
                        "result": {"type": "int", "display": "output"},
                    }
                }
            }
        }

        with patch("modiff.NodeBase._module_map", return_value=definition):
            node = NumericOptionNode("numeric-option-node")
            self.assertEqual(node(sample_rate=44100), {"result": 44100})
            self.assertEqual(node.params["sample_rate"], 44100)

    def test_structured_model_selection_preserves_typed_receipt_metadata(self):
        class ModelSelectionNode(NodeBase):
            def execute(self, model_id):
                return {"result": model_id}

        module_name = ".".join(ModelSelectionNode.__module__.split(".")[:-1])
        definition = {
            module_name: {
                "ModelSelectionNode": {
                    "params": {
                        "model_id": {
                            "type": "string",
                            "display": "modelselect",
                            "fieldOptions": {"noValidation": True, "sources": ["hub"]},
                        },
                        "result": {"type": "object", "display": "output"},
                    }
                }
            }
        }
        selection = {
            "source": "hub",
            "value": "nateraw/real-esrgan/RealESRGAN_x2plus.pth",
            "revision": "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094",
            "sha256": "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
            "byteSize": 67_061_725,
        }

        with (
            patch("modiff.NodeBase._module_map", return_value=definition),
            patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True),
        ):
            node = ModelSelectionNode("model-selection-node")
            self.assertEqual(node(model_id=selection), {"result": selection})
            self.assertIsInstance(node.params["model_id"]["byteSize"], int)

    def test_pipeline_callback_without_node_identity_preserves_diffusers_kwargs(self):
        node = _Generate()
        callback_kwargs = {"latents": object()}
        self.assertIs(node.pipe_callback(object(), 0, None, callback_kwargs), callback_kwargs)

    def test_pipeline_callback_interrupts_at_the_completed_step_boundary(self):
        node = _Generate("interrupt-test")
        node._interrupt = True
        pipe = type("Pipeline", (), {"_interrupt": False, "_num_timesteps": 30})()

        with self.assertRaisesRegex(InterruptedError, "after the current model step"):
            node.pipe_callback(pipe, 2, None, {})

        self.assertTrue(pipe._interrupt)

    def test_pipeline_callback_stops_at_the_configured_runtime_limit(self):
        node = _Generate("runtime-limit-test")
        pipe = type("Pipeline", (), {"_interrupt": False, "_num_timesteps": 30})()
        task = {
            "started_at": time.time() - 120,
            "runtimeHints": {"maxRuntimeSeconds": 60},
        }

        with (
            patch("modiff.NodeBase._server", return_value=SimpleNamespace(current_task=task)),
            self.assertRaisesRegex(TimeoutError, "configured 60 second runtime limit"),
        ):
            node.pipe_callback(pipe, 2, None, {})

        self.assertTrue(pipe._interrupt)

    def test_pipeline_callback_eta_uses_measured_step_intervals(self):
        node = _Generate("eta-test")
        pipe = type("Pipeline", (), {"_interrupt": False, "_num_timesteps": 40})()
        task = {"started_at": 1, "runtimeHints": {}}

        with (
            patch("modiff.NodeBase._server", return_value=SimpleNamespace(current_task=task)),
            patch("modiff.NodeBase.time.time", side_effect=[100.0, 700.0]),
            patch.object(node, "progress") as progress,
        ):
            node.pipe_callback(pipe, 0, None, {})
            node.pipe_callback(pipe, 1, None, {})

        first = progress.call_args_list[0]
        self.assertIsNone(first.kwargs["average_step_seconds"])
        self.assertIsNone(first.kwargs["eta_seconds"])
        second = progress.call_args_list[1]
        self.assertEqual(second.kwargs["average_step_seconds"], 600.0)
        self.assertEqual(second.kwargs["eta_seconds"], 22_800.0)

    @unittest.skipUnless(
        importlib.util.find_spec("transformers"),
        "requires the staged optional Transformers runtime",
    )
    def test_diffusers_loader_progress_reports_components_and_nested_shards(self):
        from transformers import core_model_loading
        from diffusers.utils import logging as diffusers_logging
        from transformers.utils import logging as transformers_logging

        class LoaderNode(NodeBase):
            pass

        module_name = ".".join(LoaderNode.__module__.split(".")[:-1])
        definition = {
            module_name: {
                "LoaderNode": {
                    "params": {},
                }
            }
        }
        original_tqdm = diffusers_logging.tqdm
        original_transformers_tqdm = transformers_logging.tqdm
        original_core_loading_tqdm = core_model_loading.tqdm
        with patch("modiff.NodeBase._module_map", return_value=definition):
            node = LoaderNode("loader-progress")

        with patch.object(node, "progress") as progress:
            with node.diffusers_loading_progress():
                background = threading.Thread(
                    target=lambda: list(
                        diffusers_logging.tqdm(
                            [object()],
                            desc="Background model download",
                            disable=True,
                        )
                    )
                )
                background.start()
                background.join()
                for component in diffusers_logging.tqdm(
                    [("transformer", object()), ("vae", object())],
                    desc="Loading pipeline components...",
                    disable=True,
                ):
                    if component[0] == "transformer":
                        with diffusers_logging.tqdm(
                            total=2,
                            desc="Loading checkpoint shards",
                            disable=True,
                        ) as shard_progress:
                            shard_progress.update(1)
                            shard_progress.update(1)
                        with core_model_loading.tqdm(
                            total=2,
                            desc="Loading weights",
                            disable=True,
                        ) as weight_progress:
                            weight_progress.update(1)
                            weight_progress.update(1)

        self.assertIs(diffusers_logging.tqdm, original_tqdm)
        self.assertIs(transformers_logging.tqdm, original_transformers_tqdm)
        self.assertIs(core_model_loading.tqdm, original_core_loading_tqdm)
        messages = [call.kwargs["message"] for call in progress.call_args_list]
        self.assertIn("Loading pipeline component 1/2: transformer", messages)
        self.assertIn("Loading checkpoint shards 1/2", messages)
        self.assertIn("Loading checkpoint shards 2/2", messages)
        self.assertIn("Loading weights 1/2", messages)
        self.assertIn("Loading weights 2/2", messages)
        self.assertIn("Loading pipeline component 2/2: vae", messages)
        self.assertFalse(any("Background model download" in message for message in messages))
        values = [call.args[0] for call in progress.call_args_list]
        self.assertEqual(values, sorted(values))
        self.assertLessEqual(max(values), 99)
        shard_call = next(
            call
            for call in progress.call_args_list
            if call.kwargs["message"] == "Loading weights 2/2"
        )
        self.assertEqual(shard_call.kwargs["current_step"], 2)
        self.assertEqual(shard_call.kwargs["total_steps"], 2)
        self.assertEqual(shard_call.kwargs["phase"], "shard_loading")
        component_call = next(
            call
            for call in progress.call_args_list
            if call.kwargs["message"] == "Loading pipeline component 2/2: vae"
        )
        self.assertEqual(component_call.kwargs["phase"], "component_loading")

    def test_structured_loader_progress_publishes_count_finalized_on_close(self):
        from modiff.NodeBase import _StructuredLoadingProgress

        class CloseFinalizingBar:
            desc = "Loading checkpoint shards"
            n = 4
            total = 5

            def close(self):
                self.n = self.total
                return None

        reports = []
        progress = _StructuredLoadingProgress(
            CloseFinalizingBar(),
            lambda value, message, current, total: reports.append((value, message, current, total)),
        )

        progress.close()

        self.assertEqual(reports[-1], (99, "Loading checkpoint shards 5/5", 5, 5))

    def test_structured_loader_progress_names_outer_model_component(self):
        from diffusers.utils import logging as diffusers_logging
        from modiff.NodeBase import _StructuredLoadingProgress

        reports = []
        progress = _StructuredLoadingProgress(
            diffusers_logging.tqdm(["transformer"], disable=True),
            lambda value, message, current, total: reports.append((value, message, current, total)),
            description="Loading model components",
            total=1,
        )

        list(progress)

        self.assertIn((0, "Loading model component 1/1: transformer", 1, 1), reports)

    def test_nested_audio_arrays_compare_without_image_attributes(self):
        left = {"audio": {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}}
        right = {"audio": {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}}

        self.assertTrue(deep_equal(left, right))

        right["audio"]["samples"][0, 10] = 0.5
        self.assertFalse(deep_equal(left, right))

    def test_direct_node_base_imports_preserve_complete_module_registry(self):
        script = """
import json
from modiff.NodeBase import NodeBase
import modules
print(json.dumps({
    "module_count": len(modules.MODULE_MAP),
    "node_count": modules.total_nodes,
    "recomputed_node_count": sum(len(nodes) for nodes in modules.MODULE_MAP.values()),
    "module_names": sorted(modules.MODULE_MAP),
}))
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
        self.assertEqual(payload["module_count"], len(payload["module_names"]))
        self.assertEqual(payload["node_count"], payload["recomputed_node_count"])
        self.assertGreater(payload["node_count"], 0)
        self.assertTrue(
            {"modules.DiffusersImage", "modules.ModularDiffusers"}.issubset(payload["module_names"])
        )


if __name__ == "__main__":
    unittest.main()
