import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from modiff.NodeBase import NodeBase
from modiff.server import WebServer
from modiff.workflow_store import (
    delete_workflow,
    get_workflow,
    list_workflow_summaries,
    list_workflows,
    save_workflow,
)
from modules.ModularDiffusers.custom_pipeline import CustomPipelineContractError


class FakeRequest:
    def __init__(self, workflow_id, payload=None, query=None):
        self.match_info = {"workflow_id": workflow_id, "task_id": workflow_id, "output_id": workflow_id}
        self._payload = payload
        self.query = query or {}

    async def json(self):
        return self._payload


class WorkflowStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directory.cleanup()

    def test_atomic_saved_workflow_revisions_and_delete(self):
        first = save_workflow(self.directory.name, "workflow-1", {"title": "One", "snapshot": {"nodes": []}})
        second = save_workflow(self.directory.name, "workflow-1", {"title": "Two", "snapshot": {"nodes": [1]}})

        self.assertEqual(first["revision"], 1)
        self.assertEqual(second["revision"], 2)
        self.assertEqual(get_workflow(self.directory.name, "workflow-1")["title"], "Two")
        self.assertEqual([item["id"] for item in list_workflows(self.directory.name)], ["workflow-1"])
        summaries = list_workflow_summaries(self.directory.name)
        self.assertEqual([item["id"] for item in summaries], ["workflow-1"])
        self.assertNotIn("snapshot", summaries[0])
        self.assertTrue(delete_workflow(self.directory.name, "workflow-1"))
        self.assertIsNone(get_workflow(self.directory.name, "workflow-1"))

    def test_workflow_id_cannot_escape_backend_storage(self):
        with self.assertRaisesRegex(ValueError, "Workflow id"):
            save_workflow(self.directory.name, "../escape", {"snapshot": {}})

    def test_summary_listing_reuses_only_unchanged_metadata_and_detects_external_replacement(self):
        from modiff import workflow_store

        save_workflow(self.directory.name, "one", {"title": "First", "snapshot": {"nodes": [1]}})
        list_workflow_summaries(self.directory.name)
        with patch.object(workflow_store, "_read", wraps=workflow_store._read) as read:
            result = list_workflow_summaries(self.directory.name)
            result[0]["title"] = "Not a saved edit"
            self.assertEqual(list_workflow_summaries(self.directory.name)[0]["title"], "First")
            read.assert_not_called()

        path = Path(self.directory.name) / "user-workflows/one.json"
        record = get_workflow(self.directory.name, "one")
        record["title"] = "External replacement"
        replacement = path.with_suffix(".replacement")
        replacement.write_text(json.dumps(record), encoding="utf-8")
        replacement.replace(path)
        self.assertEqual(list_workflow_summaries(self.directory.name)[0]["title"], "External replacement")
        path.write_text("broken json", encoding="utf-8")
        self.assertEqual(list_workflow_summaries(self.directory.name), [])
        path.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(list_workflow_summaries(self.directory.name)[0]["title"], "External replacement")
        delete_workflow(self.directory.name, "one")
        self.assertEqual(list_workflow_summaries(self.directory.name), [])
        save_workflow(self.directory.name, "one", {"title": "Recreated", "snapshot": {}})
        self.assertEqual(list_workflow_summaries(self.directory.name)[0]["title"], "Recreated")

    async def test_workflow_storage_and_response_serialization_do_not_block_http_loop(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        messages = []
        server.queue_message = messages.append
        loop_thread = threading.get_ident()
        record = {"id": "one", "snapshot": {}, "revision": 1}
        for handler_name, store_name, stored, query in (
            ("workflows_list", "list_workflows", [record], {}),
            ("workflows_list", "list_workflow_summaries", [{"id": "one"}], {"view": "summary"}),
            ("workflow_get", "get_workflow", record, {}),
            ("workflow_put", "save_workflow", record, {}),
            ("workflow_delete", "delete_workflow", True, {}),
        ):
            with self.subTest(handler=handler_name, store=store_name):
                heartbeat = threading.Event()
                observed = []

                def storage(*_args, **_kwargs):
                    observed.append(heartbeat.wait(timeout=0.2))
                    self.assertNotEqual(threading.get_ident(), loop_thread)
                    return stored

                def serialize(payload):
                    self.assertNotEqual(threading.get_ident(), loop_thread)
                    return json.dumps(payload).encode("utf-8")

                timer = asyncio.get_running_loop().call_later(0.01, heartbeat.set)
                try:
                    with (
                        patch(f"modiff.workflow_store.{store_name}", side_effect=storage),
                        patch.object(server, "_json_response_bytes", side_effect=serialize),
                    ):
                        response = await getattr(server, handler_name)(FakeRequest("one", {"snapshot": {}}, query))
                        self.assertEqual(response.status, 200)
                finally:
                    timer.cancel()
                    heartbeat.set()
                self.assertEqual(observed, [True])

    def test_summary_cache_is_bounded_and_never_retains_graph_snapshots(self):
        from modiff import workflow_store

        with (
            patch.object(workflow_store, "_SUMMARY_CACHE", workflow_store.OrderedDict()),
            patch.object(workflow_store, "_SUMMARY_CACHE_LIMIT", 2),
        ):
            for index in range(3):
                save_workflow(self.directory.name, str(index), {"snapshot": {"privateGraph": "not metadata"}})
            self.assertEqual(len(list_workflow_summaries(self.directory.name)), 3)
            self.assertEqual(len(workflow_store._SUMMARY_CACHE), 2)
            self.assertNotIn("privateGraph", repr(workflow_store._SUMMARY_CACHE))
            self.assertNotIn("snapshot", repr(workflow_store._SUMMARY_CACHE))

    async def test_cancelled_save_request_still_notifies_after_durable_write(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        messages = []
        notified = asyncio.Event()
        entered = threading.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def notify(message):
            messages.append(message)
            loop.call_soon_threadsafe(notified.set)

        def save(*args):
            entered.set()
            release.wait(timeout=2)
            return save_workflow(*args)

        server.queue_message = notify
        with patch("modiff.workflow_store.save_workflow", side_effect=save):
            task = asyncio.create_task(server.workflow_put(FakeRequest("cancelled", {"snapshot": {"nodes": []}})))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
            await asyncio.wait_for(notified.wait(), timeout=2)
        self.assertEqual(get_workflow(self.directory.name, "cancelled")["revision"], 1)
        self.assertEqual(messages[0]["type"], "workflow_updated")

    async def test_routes_broadcast_backend_workflow_updates(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        messages = []
        server.queue_message = messages.append

        response = await server.workflow_put(
            FakeRequest("shared", {"title": "Shared", "snapshot": {"nodes": [], "edges": []}, "clientId": "tab-a"})
        )
        record = json.loads(response.text)
        self.assertEqual(record["revision"], 1)
        self.assertEqual(messages[-1]["type"], "workflow_updated")

        listing = json.loads((await server.workflows_list(None)).text)
        self.assertEqual(listing["workflows"][0]["id"], "shared")
        summary_listing = json.loads((await server.workflows_list(FakeRequest("", query={"view": "summary"}))).text)
        self.assertNotIn("snapshot", summary_listing["workflows"][0])

        await server.workflow_delete(FakeRequest("shared"))
        self.assertEqual(messages[-1], {"type": "workflow_deleted", "workflow_id": "shared"})

    async def test_run_snapshot_retains_workflow_identity_after_frontend_disconnect(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        server.queue_message = lambda *_args, **_kwargs: None
        graph = {
            "nodes": [],
            "runtimeHints": {
                "clientRunId": "client-run-a",
                "workflowTabId": "workflow-a",
                "workflowTitle": "Shared workflow",
                "workflowSnapshot": {"nodes": [], "edges": []},
            },
        }
        task_id = await server.queue_task(lambda *_args: None, (graph,), None, "closed-tab", name="Graph execution")
        server.current_task = {
            "task_id": task_id,
            "name": "Graph execution",
            "sid": "closed-tab",
            "started_at": 1,
            "runtimeHints": graph["runtimeHints"],
        }
        server.queued_tasks.pop(task_id, None)
        server._record_terminal_task("completed")
        recent = server.recent_tasks[0]
        self.assertEqual(recent["workflow_tab_id"], "workflow-a")
        self.assertEqual(recent["workflow_title"], "Shared workflow")
        self.assertEqual(recent["workflow_snapshot"], {"nodes": [], "edges": []})

        queue_payload = json.loads((await server.get_queue(None)).text)
        self.assertNotIn("workflow_snapshot", queue_payload["recent"][0])
        self.assertTrue(queue_payload["recent"][0]["has_workflow_snapshot"])

        response = await server.get_run(FakeRequest(task_id))
        payload = json.loads(response.text)
        self.assertEqual(payload["workflow_id"], "workflow-a")
        self.assertEqual(payload["workflow_snapshot"], {"nodes": [], "edges": []})
        self.assertEqual(payload["outputs"], [])

    async def test_queued_and_early_running_snapshots_retain_submission_identity(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        server.loop = asyncio.get_running_loop()
        messages = []
        server.queue_message = messages.append
        graph = {
            "nodes": {},
            "paths": [],
            "runtimeHints": {
                "clientRunId": "queued-client",
                "runInputHash": "queued-hash",
                "workflowTabId": "queued-workflow",
                "nodeId": "preview-node",
                "autoResourceCandidateId": "queued-auto-candidate",
            },
        }
        future = asyncio.get_running_loop().create_future()
        task_id = await server.queue_task(
            lambda _graph: "finished",
            (graph,),
            future,
            "queued-session",
            name="Graph execution",
        )

        queued_payload = json.loads((await server.get_queue(None)).text)["queued"][task_id]
        self.assertEqual(queued_payload["client_run_id"], "queued-client")
        self.assertEqual(queued_payload["run_input_hash"], "queued-hash")
        self.assertEqual(queued_payload["workflow_tab_id"], "queued-workflow")
        self.assertEqual(queued_payload["node_id"], "preview-node")

        worker = asyncio.create_task(server._main_worker())
        self.assertEqual(await asyncio.wait_for(future, timeout=2), "finished")
        await asyncio.wait_for(server.main_queue.join(), timeout=2)
        server._shutdown_event.set()
        await server.main_queue.put(None)
        await asyncio.wait_for(worker, timeout=2)

        started = next(message for message in messages if message.get("type") == "task_started")
        self.assertEqual(started["client_run_id"], "queued-client")
        self.assertEqual(started["workflow_tab_id"], "queued-workflow")
        self.assertEqual(started["current"]["run_input_hash"], "queued-hash")
        self.assertEqual(started["current"]["resourceCandidateId"], "queued-auto-candidate")
        completed = next(message for message in messages if message.get("type") == "task_completed")
        self.assertEqual(completed["client_run_id"], "queued-client")
        self.assertEqual(completed["workflow_tab_id"], "queued-workflow")

    async def test_queued_field_action_echoes_workflow_canvas_ownership_on_mutations_and_completion(self):
        class FieldNode(NodeBase):
            def refresh(self, _values, _ref):
                self.set_field_params("dtype", {"options": ["float16", "bfloat16"]})

        FieldNode.__module__ = "tests.test_workflow_store"
        module_name = "tests"
        definition = {
            module_name: {
                "FieldNode": {
                    "params": {
                        "dtype": {"onChange": "refresh"},
                    }
                }
            }
        }
        with patch("modiff.NodeBase._module_map", return_value=definition):
            node = FieldNode("field-node")

        server = WebServer(modules=definition, work_dir=self.directory.name, data_dir=self.directory.name)
        server.loop = asyncio.get_running_loop()
        server.node_cache["field-node"] = node
        messages = []
        server.queue_message = lambda message, *_args, **_kwargs: messages.append(message)
        request = FakeRequest(
            "field-node",
            {
                "node": "field-node",
                "sid": "field-session",
                "module": module_name,
                "action": "FieldNode",
                "fn": "refresh",
                "values": {"dtype": "float16"},
                "fieldKey": "dtype",
                "queue": True,
                "workflowTabId": "workflow-field",
                "workflowCanvasEpoch": 17,
                "workflowFormEpoch": 23,
            },
        )

        with patch("modiff.NodeBase._server", return_value=server):
            response = json.loads((await server.field_action(request)).text)
            worker = asyncio.create_task(server._main_worker())
            await asyncio.wait_for(server.main_queue.join(), timeout=2)
            server._shutdown_event.set()
            await server.main_queue.put(None)
            await asyncio.wait_for(worker, timeout=2)

        task_id = response["task_id"]
        field_message = next(message for message in messages if message.get("type") == "set_field_params")
        self.assertEqual(field_message["task_id"], task_id)
        self.assertEqual(field_message["workflow_tab_id"], "workflow-field")
        self.assertEqual(field_message["workflow_canvas_epoch"], 17)
        self.assertEqual(field_message["workflow_form_epoch"], 23)
        self.assertEqual(field_message["sid"], "field-session")

        completed = next(message for message in messages if message.get("type") == "task_completed")
        self.assertEqual(completed["task_id"], task_id)
        self.assertEqual(completed["workflow_tab_id"], "workflow-field")
        self.assertEqual(completed["workflow_canvas_epoch"], 17)
        self.assertEqual(completed["workflow_form_epoch"], 23)
        self.assertEqual(completed["args"][1]["node"], "field-node")

    async def test_synchronous_canvas_scoped_field_action_echoes_tab_and_canvas_without_form_epoch(self):
        class FieldNode(NodeBase):
            def refresh(self, _values, _ref):
                self.set_field_params("dtype", {"options": ["float16", "bfloat16"]})

        FieldNode.__module__ = "tests.test_workflow_store"
        module_name = "tests"
        definition = {
            module_name: {
                "FieldNode": {
                    "params": {
                        "dtype": {"onChange": "refresh"},
                    }
                }
            }
        }
        with patch("modiff.NodeBase._module_map", return_value=definition):
            node = FieldNode("canvas-field-node")

        server = WebServer(modules=definition, work_dir=self.directory.name, data_dir=self.directory.name)
        server.loop = asyncio.get_running_loop()
        server.node_cache["canvas-field-node"] = node
        messages = []
        server.queue_message = lambda message, *_args, **_kwargs: messages.append(message)
        request = FakeRequest(
            "canvas-field-node",
            {
                "node": "canvas-field-node",
                "sid": "canvas-field-session",
                "module": module_name,
                "action": "FieldNode",
                "fn": "refresh",
                "values": {"dtype": "bfloat16"},
                "fieldKey": "dtype",
                "queue": False,
                "workflowTabId": "workflow-canvas-field",
                "workflowCanvasEpoch": 31,
            },
        )

        with patch("modiff.NodeBase._server", return_value=server):
            response = await server.field_action(request)

        payload = json.loads(response.text)
        self.assertEqual(response.status, 200)
        self.assertFalse(payload["error"])
        field_message = next(message for message in messages if message.get("type") == "set_field_params")
        self.assertEqual(field_message["workflow_tab_id"], "workflow-canvas-field")
        self.assertEqual(field_message["workflow_canvas_epoch"], 31)
        self.assertNotIn("workflow_form_epoch", field_message)
        self.assertEqual(field_message["sid"], "canvas-field-session")

    async def test_field_action_rejects_unknown_or_undeclared_targets_before_import(self):
        module_name = "modules.ModularDiffusers"
        action_name = "ModelsLoader"
        definition = {
            module_name: {
                action_name: {
                    "params": {
                        "repo_id": {"onChange": "refresh_pipeline_identity"},
                    }
                }
            }
        }
        server = WebServer(modules=definition, work_dir=self.directory.name, data_dir=self.directory.name)
        base_payload = {
            "node": "imported-loader",
            "sid": "field-session",
            "module": module_name,
            "action": action_name,
            "fieldKey": "repo_id",
            "fn": "refresh_pipeline_identity",
            "values": {},
            "queue": False,
        }
        cases = {
            "unknown module": {"module": "custom.Attacker"},
            "unknown action": {"action": "AttackerNode"},
            "unknown field": {"fieldKey": "removed_or_imported_field"},
            "undeclared method": {"fn": "prepare_for_workflow_reuse"},
        }

        with patch("modiff.server.import_module") as import_mock:
            for label, override in cases.items():
                with self.subTest(label=label):
                    response = await server.field_action(
                        FakeRequest("imported-loader", {**base_payload, **override})
                    )
                    payload = json.loads(response.text)
                    self.assertEqual(response.status, 400)
                    self.assertTrue(payload["error"])
            import_mock.assert_not_called()

    async def test_field_action_rejects_non_object_payload_before_dispatch(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        with patch("modiff.server.import_module") as import_mock:
            for payload in (None, [], ["attacker"]):
                with self.subTest(payload=payload):
                    response = await server.field_action(FakeRequest("field-node", payload))
                    body = json.loads(response.text)
                    self.assertEqual(response.status, 400)
                    self.assertTrue(body["error"])
                    self.assertIn("JSON object", body["message"])
        import_mock.assert_not_called()

    async def test_field_action_dispatches_authorized_models_loader_callback(self):
        module_name = "modules.ModularDiffusers"
        action_name = "ModelsLoader"
        definition = {
            module_name: {
                action_name: {
                    "params": {
                        "repo_id": {
                            "onSignal": [
                                {"action": "value", "data": "repo_id"},
                                [{"action": "exec", "data": "refresh_pipeline_identity"}],
                            ]
                        },
                    }
                }
            }
        }

        class CachedModelsLoader:
            module_name = "modules.ModularDiffusers"
            class_name = "ModelsLoader"

            def __init__(self):
                self._sid = None
                self.calls = []

            def refresh_pipeline_identity(self, values, ref):
                self.calls.append((values, ref))

            def prepare_for_workflow_reuse(self):
                raise AssertionError("An undeclared callback was dispatched.")

        cached_node = CachedModelsLoader()
        server = WebServer(modules=definition, work_dir=self.directory.name, data_dir=self.directory.name)
        server.loop = asyncio.get_running_loop()
        server.node_cache["models-loader"] = cached_node

        with patch(
            "modiff.server.field_action_optional_runtime_requirement",
            return_value={
                "schemaVersion": 1,
                "delivery": "optional_overlay",
                "requiredNow": True,
                "profileIds": ["huggingface-transformers-peft-5.14.1-0.20.0"],
                "executionProfileIds": ["qwen-image:modular"],
                "state": "active",
                "reason": "optional_runtime_active",
            },
        ):
            response = await server.field_action(
                FakeRequest(
                    "models-loader",
                    {
                        "node": "models-loader",
                        "sid": "field-session",
                        "module": module_name,
                        "action": action_name,
                        "fieldKey": "repo_id",
                        "fn": "refresh_pipeline_identity",
                        "values": {
                            "model_type": "QwenImageModularPipeline",
                            "repo_id": {"source": "hub", "value": "Qwen/Qwen-Image-2512"},
                        },
                        "queue": False,
                    },
                )
            )

        payload = json.loads(response.text)
        self.assertEqual(response.status, 200)
        self.assertFalse(payload["error"])
        self.assertEqual(payload["ref"], {"node": "models-loader", "key": "repo_id", "queue": False})
        self.assertEqual(cached_node._sid, "field-session")
        self.assertEqual(
            cached_node.calls,
            [
                (
                    {
                        "model_type": "QwenImageModularPipeline",
                        "repo_id": {"source": "hub", "value": "Qwen/Qwen-Image-2512"},
                    },
                    {"node": "models-loader", "key": "repo_id", "queue": False},
                )
            ],
        )

    async def test_field_action_preserves_actionable_custom_pipeline_error_contract(self):
        module_name = "modules.ModularDiffusers"
        action_name = "ModelsLoader"
        definition = {
            module_name: {
                action_name: {
                    "params": {"repo_id": {"onChange": "refresh_pipeline_identity"}},
                }
            }
        }

        class CachedModelsLoader:
            module_name = "modules.ModularDiffusers"
            class_name = "ModelsLoader"
            _sid = None

            def refresh_pipeline_identity(self, _values, _ref):
                raise CustomPipelineContractError(
                    "custom_pipeline_unpinned_auxiliary",
                    "The auxiliary repository is mutable.",
                    "Pin every auxiliary repository to an exact commit.",
                )

        server = WebServer(modules=definition, work_dir=self.directory.name, data_dir=self.directory.name)
        server.loop = asyncio.get_running_loop()
        server.node_cache["models-loader"] = CachedModelsLoader()
        request = FakeRequest(
            "models-loader",
            {
                "node": "models-loader",
                "sid": "field-session",
                "module": module_name,
                "action": action_name,
                "fieldKey": "repo_id",
                "fn": "refresh_pipeline_identity",
                "values": {"model_type": "DummyCustomPipeline"},
                "queue": False,
            },
        )
        active_requirement = {
            "schemaVersion": 1,
            "delivery": "optional_overlay",
            "requiredNow": True,
            "profileIds": ["huggingface-transformers-peft-5.14.1-0.20.0"],
            "executionProfileIds": ["custom-modular:reviewed-loader"],
            "state": "active",
            "reason": "optional_runtime_active",
        }
        with patch("modiff.server.field_action_optional_runtime_requirement", return_value=active_requirement):
            response = await server.field_action(request)

        payload = json.loads(response.text)
        self.assertEqual(response.status, 409)
        self.assertEqual(payload["category"], "custom_pipeline")
        self.assertEqual(payload["error_code"], "custom_pipeline_unpinned_auxiliary")
        self.assertEqual(payload["recovery_hint"], "Pin every auxiliary repository to an exact commit.")

    async def test_generated_media_is_preserved_without_a_frontend_history_post(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        runtime_hints = {
            "clientRunId": "disconnected-client",
            "runInputHash": "disconnected-hash",
            "workflowTabId": "disconnected-workflow",
            "workflowSnapshot": {
                "nodes": [],
                "edges": [],
                "studioForm": {
                    "mode": "text_to_image",
                    "modelType": "QwenImageModularPipeline",
                    "prompt": "Preserve this output",
                    "modelRepo": "Qwen/Qwen-Image-2512",
                },
            },
        }
        server.current_task = {
            "task_id": "disconnected-task",
            "name": "Graph execution",
            "sid": "gone-browser",
            "started_at": 1,
            "attempt_index": 0,
            "runtimeHints": runtime_hints,
        }
        server.task_graphs["disconnected-task"] = {"runtimeHints": runtime_hints}
        one_pixel_png = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
            "AScY42YAAAAASUVORK5CYII="
        )

        output_id, persisted = server._persist_generated_output_update(
            {
                "type": "update_value",
                "task_id": "disconnected-task",
                "client_run_id": "disconnected-client",
                "run_input_hash": "disconnected-hash",
                "workflow_tab_id": "disconnected-workflow",
                "attempt_index": 0,
                "node": "preview",
                "key": "images",
                "data_type": "image",
                "value": [one_pixel_png],
            },
            display="ui_image",
        )
        self.assertTrue(persisted)
        self.assertTrue(output_id.startswith("run-output-"))
        server._record_terminal_task("completed")
        server.current_task = None

        payload = json.loads((await server.get_run(FakeRequest("disconnected-task"))).text)
        self.assertEqual(len(payload["outputs"]), 1)
        output = payload["outputs"][0]
        self.assertEqual(output["id"], output_id)
        self.assertEqual(output["taskId"], "disconnected-task")
        self.assertEqual(output["clientRunId"], "disconnected-client")
        self.assertEqual(output["workflowTabId"], "disconnected-workflow")
        self.assertTrue(output["url"].startswith("/file?file="))
        self.assertEqual(output["displayType"], "image")
        self.assertEqual(output["repo"], "Qwen/Qwen-Image-2512")
        self.assertNotIn("modelRepo", runtime_hints)
        self.assertNotIn("source", runtime_hints)
        self.assertTrue(output["backendMediaPath"].startswith("@data/"))
        self.assertTrue(server._resolve_managed_path_identifier(output["backendMediaPath"]).is_file())

    async def test_generated_cluster_media_uses_the_qualified_cluster_form(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        runtime_hints = {
            "source": "hugging-face-cluster",
            "modelType": "ErnieImageModularPipeline",
            "modelName": "Ernie Image — Text To Image",
            "resolvedArtifact": "baidu/ERNIE-Image-Turbo",
            "optimizationQualificationForm": {
                "mode": "text_to_image",
                "modelType": "ErnieImageModularPipeline",
                "prompt": "The exact Cluster prompt",
                "modelRepo": "fixture/stale-form-repository",
                "steps": 1,
                "width": 1024,
                "height": 1024,
            },
            "workflowSnapshot": {
                "nodes": [],
                "edges": [],
                "studioForm": {
                    "mode": "text_to_image",
                    "modelType": "ZImageModularPipeline",
                    "prompt": "Stale Studio form",
                    "steps": 8,
                },
            },
        }
        server.current_task = {
            "task_id": "cluster-task",
            "name": "Graph execution",
            "sid": "cluster-browser",
            "started_at": 1,
            "attempt_index": 0,
            "runtimeHints": runtime_hints,
        }
        server.task_graphs["cluster-task"] = {"runtimeHints": runtime_hints}

        _output_id, persisted = server._persist_generated_output_update(
            {
                "type": "update_value",
                "task_id": "cluster-task",
                "client_run_id": "cluster-client",
                "run_input_hash": "cluster-hash",
                "attempt_index": 0,
                "node": "preview",
                "key": "images",
                "value": ["data:image/png;base64,iVBORw0KGgo="],
            },
            display="ui_image",
        )

        self.assertTrue(persisted)
        output = server._studio_outputs_for_run("cluster-task")[0]
        self.assertEqual(output["modelType"], "ErnieImageModularPipeline")
        self.assertEqual(output["repo"], "baidu/ERNIE-Image-Turbo")
        self.assertEqual(output["prompt"], "The exact Cluster prompt")
        self.assertEqual(output["steps"], 1)
        self.assertEqual(output["formSnapshot"], runtime_hints["optimizationQualificationForm"])

    async def test_preview_slot_changes_only_on_admission_and_output_promotion(self):
        modules = {
            "modules.Diffusers": {
                "Preview": {
                    "params": {
                        "images": {"display": "ui_image"},
                    }
                }
            }
        }
        server = WebServer(modules=modules, work_dir=self.directory.name, data_dir=self.directory.name)
        server.queue_message = lambda *_args, **_kwargs: None

        def graph(client_run_id):
            return {
                "nodes": {
                    "preview": {
                        "module": "modules.Diffusers",
                        "action": "Preview",
                        "params": {"images": {"sourceId": "generate", "sourceKey": "images"}},
                    }
                },
                "runtimeHints": {
                    "clientRunId": client_run_id,
                    "workflowTabId": "workflow-a",
                },
            }

        first_graph = graph("client-a")
        first_task_id = await server.queue_task(
            lambda *_args: None, (first_graph,), None, "session", name="Graph execution"
        )
        state = server._read_studio_output_state()
        slot = next(iter(state["previewSlots"].values()))
        self.assertEqual(slot["status"], "pending")
        self.assertIsNone(slot["currentOutputId"])
        self.assertEqual(slot["pendingTaskId"], first_task_id)

        server.current_task = {
            "task_id": first_task_id,
            "name": "Graph execution",
            "sid": "session",
            "attempt_index": 0,
            "runtimeHints": first_graph["runtimeHints"],
        }
        output_id, persisted = server._persist_generated_output_update(
            {
                "type": "update_value",
                "task_id": first_task_id,
                "client_run_id": "client-a",
                "workflow_tab_id": "workflow-a",
                "attempt_index": 0,
                "node": "preview",
                "key": "images",
                "value": ["data:image/png;base64,iVBORw0KGgo="],
            },
            display="ui_image",
        )
        self.assertTrue(persisted)
        state = server._read_studio_output_state()
        slot = next(iter(state["previewSlots"].values()))
        self.assertEqual(slot["status"], "ready")
        self.assertEqual(slot["currentOutputId"], output_id)
        self.assertIsNone(slot["pendingTaskId"])

        second_graph = graph("client-b")
        second_task_id = await server.queue_task(
            lambda *_args: None, (second_graph,), None, "session", name="Graph execution"
        )
        state = server._read_studio_output_state()
        slot = next(iter(state["previewSlots"].values()))
        self.assertEqual(slot["status"], "pending")
        self.assertIsNone(slot["currentOutputId"])
        self.assertEqual(slot["pendingTaskId"], second_task_id)
        self.assertEqual([output["id"] for output in state["outputs"]], [output_id])

        server._mark_studio_preview_run_terminal(second_task_id, "completed")
        state = server._read_studio_output_state()
        slot = next(iter(state["previewSlots"].values()))
        self.assertEqual(slot["status"], "completed_without_output")
        self.assertIsNone(slot["currentOutputId"])
        self.assertEqual([output["id"] for output in state["outputs"]], [output_id])

    async def test_deleting_current_preview_never_promotes_an_older_output(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        outputs = [
            {
                "id": "new",
                "workflowTabId": "workflow-a",
                "nodeId": "preview",
                "fieldKey": "images",
                "createdAt": 2,
            },
            {
                "id": "old",
                "workflowTabId": "workflow-a",
                "nodeId": "preview",
                "fieldKey": "images",
                "createdAt": 1,
            },
        ]
        key = server._studio_preview_slot_key("workflow-a", "preview", "images")
        slot = {
            "schemaVersion": 1,
            "workflowTabId": "workflow-a",
            "nodeId": "preview",
            "fieldKey": "images",
            "currentOutputId": "new",
            "pendingClientRunId": None,
            "pendingTaskId": None,
            "generation": 2,
            "attemptIndex": 0,
            "status": "ready",
            "updatedAt": 2,
        }
        server._write_studio_output_state(outputs, {key: slot}, revision=2)
        await server.studio_outputs_delete(FakeRequest("new"))

        restored = server._read_studio_output_state()
        self.assertEqual([output["id"] for output in restored["outputs"]], ["old"])
        self.assertIsNone(restored["previewSlots"][key]["currentOutputId"])
        self.assertEqual(restored["previewSlots"][key]["status"], "empty")

    async def test_terminal_failure_retains_actionable_recovery_metadata(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        server.current_task = {
            "task_id": "failed-run",
            "name": "Graph execution",
            "sid": "closed-tab",
            "started_at": 1,
            "runtimeHints": {
                "clientRunId": "failed-client",
                "runInputHash": "failed-hash",
                "workflowTabId": "failed-workflow",
            },
        }
        server.task_graphs["failed-run"] = {"runtimeHints": server.current_task["runtimeHints"]}
        server._record_terminal_task(
            "failed",
            error_payload={
                "message": "CUDA out of memory while loading",
                "exception_type": "OutOfMemoryError",
                "category": "oom",
                "error_code": "cuda_oom",
                "recovery_hint": "Apply the Low-VRAM preset and retry.",
                "node": "loader-node",
                "node_name": "Load Models",
                "oom": True,
            },
        )
        server.current_task = None

        payload = json.loads((await server.get_run(FakeRequest("failed-run"))).text)
        task = payload["task"]
        self.assertEqual(task["client_run_id"], "failed-client")
        self.assertEqual(task["workflow_tab_id"], "failed-workflow")
        self.assertEqual(task["run_input_hash"], "failed-hash")
        self.assertEqual(task["exception_type"], "OutOfMemoryError")
        self.assertEqual(task["category"], "oom")
        self.assertEqual(task["error_code"], "cuda_oom")
        self.assertEqual(task["recovery_hint"], "Apply the Low-VRAM preset and retry.")
        self.assertEqual(task["node"], "loader-node")
        self.assertEqual(task["node_name"], "Load Models")
        self.assertTrue(task["oom"])

    async def test_run_details_return_only_strictly_correlated_outputs(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        server.queue_message = lambda *_args, **_kwargs: None
        graph = {
            "nodes": [],
            "runtimeHints": {
                "clientRunId": "client-run-a",
                "workflowTabId": "workflow-a",
                "workflowTitle": "Shared workflow",
                "workflowSnapshot": {"nodes": [], "edges": []},
            },
        }
        task_id = await server.queue_task(lambda *_args: None, (graph,), None, "closed-tab", name="Graph execution")
        server.current_task = {
            "task_id": task_id,
            "name": "Graph execution",
            "sid": "closed-tab",
            "started_at": 1,
            "runtimeHints": graph["runtimeHints"],
        }
        server.queued_tasks.pop(task_id, None)
        server._record_terminal_task("completed")
        server._write_studio_outputs(
            [
                {
                    "id": "direct-match",
                    "taskId": task_id,
                    "clientRunId": "client-run-a",
                    "url": "/file?file=direct.webp",
                },
                {
                    "id": "provenance-match",
                    "provenance": {
                        "backendExecutionId": task_id,
                        "clientRunId": "client-run-a",
                    },
                    "url": "/file?file=provenance.webp",
                },
                {
                    "id": "legacy-task-match",
                    "taskId": task_id,
                    "url": "/file?file=legacy.webp",
                },
                {
                    "id": "media-item-match",
                    "mediaItems": [
                        {
                            "taskId": task_id,
                            "clientRunId": "client-run-a",
                            "url": "/file?file=item.webp",
                        }
                    ],
                    "url": "/file?file=item.webp",
                },
                {
                    "id": "wrong-task",
                    "taskId": "another-task",
                    "clientRunId": "client-run-a",
                    "url": "/file?file=wrong-task.webp",
                },
                {
                    "id": "wrong-client",
                    "taskId": task_id,
                    "clientRunId": "client-run-b",
                    "url": "/file?file=wrong-client.webp",
                },
                {
                    "id": "workflow-only",
                    "workflowTabId": "workflow-a",
                    "url": "/file?file=workflow-only.webp",
                },
                {
                    "id": "conflicting-task-identities",
                    "taskId": "another-task",
                    "backendProvenance": {
                        "backendExecutionId": task_id,
                        "clientRunId": "client-run-a",
                    },
                    "url": "/file?file=conflict.webp",
                },
            ]
        )

        payload = json.loads((await server.get_run(FakeRequest(task_id))).text)

        self.assertEqual(
            [output["id"] for output in payload["outputs"]],
            ["direct-match", "provenance-match", "legacy-task-match", "media-item-match"],
        )

    async def test_run_details_use_terminal_task_client_identity_without_a_saved_graph(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        server.recent_tasks = [
            {
                "task_id": "remote-task",
                "name": "Direct render",
                "status": "completed",
                "client_run_id": "remote-client-a",
            }
        ]
        server._write_studio_outputs(
            [
                {
                    "id": "remote-match",
                    "taskId": "remote-task",
                    "clientRunId": "remote-client-a",
                    "url": "/file?file=remote.webp",
                },
                {
                    "id": "remote-wrong-client",
                    "taskId": "remote-task",
                    "clientRunId": "remote-client-b",
                    "url": "/file?file=wrong-client.webp",
                },
            ]
        )

        payload = json.loads((await server.get_run(FakeRequest("remote-task"))).text)

        self.assertEqual([output["id"] for output in payload["outputs"]], ["remote-match"])

    async def test_run_details_recover_workflow_navigation_from_durable_output_after_restart(self):
        server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        server.recent_tasks = [
            {
                "task_id": "restarted-task",
                "name": "Graph execution",
                "status": "completed",
                "client_run_id": "restarted-client",
                "workflow_tab_id": "workflow-after-restart",
            }
        ]
        workflow_snapshot = {
            "nodes": [{"id": "preview"}],
            "edges": [],
            "viewport": {"x": 0, "y": 0, "zoom": 1},
        }
        server._write_studio_outputs(
            [
                {
                    "id": "restarted-output",
                    "taskId": "restarted-task",
                    "clientRunId": "restarted-client",
                    "workflowTabId": "workflow-after-restart",
                    "url": "/file?file=restarted.webp",
                    "apiGraphSnapshot": {
                        "runtimeHints": {
                            "clientRunId": "restarted-client",
                            "workflowTabId": "workflow-after-restart",
                            "workflowTitle": "Recovered workflow name",
                            "workflowSnapshot": workflow_snapshot,
                        }
                    },
                }
            ]
        )

        payload = json.loads((await server.get_run(FakeRequest("restarted-task"))).text)

        self.assertEqual(payload["workflow_id"], "workflow-after-restart")
        self.assertEqual(payload["workflow_title"], "Recovered workflow name")
        self.assertEqual(payload["workflow_snapshot"], workflow_snapshot)


if __name__ == "__main__":
    unittest.main()
