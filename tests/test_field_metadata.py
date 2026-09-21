"""Authoring callbacks must not borrow a running graph's model owners."""

import asyncio
import json
import importlib.util
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from modiff.server import WebServer
from modiff.field_metadata import is_metadata_field_action, metadata_field_callback
from modiff.runtime_overlays import OverlayInstallBusy


class FieldMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.server = WebServer(
            modules={"modules.ModularDiffusers": {"ModelsLoader": {
                "params": {"model_type": {"onChange": "set_filters"}},
            }}}, work_dir=directory.name, data_dir=directory.name,
        )
        self.server.loop = asyncio.get_running_loop()

    def request(self, **updates):
        data = {
            "node": "same-model-owner", "sid": "editing-session",
            "module": "modules.ModularDiffusers", "action": "ModelsLoader",
            "fieldKey": "model_type", "fn": "set_filters",
            "values": {"model_type": ""}, "queue": False,
            "workflowTabId": "next-draft", "workflowCanvasEpoch": 3,
            "workflowFormEpoch": 7,
        }
        data.update(updates)
        return SimpleNamespace(json=AsyncMock(return_value=data))

    async def test_loader_fields_complete_while_same_id_model_owner_is_locked(self):
        owner = SimpleNamespace(
            module_name="modules.ModularDiffusers", class_name="ModelsLoader",
            _sid="running-session", set_filters=Mock(side_effect=AssertionError("live owner touched")),
        )
        self.server.node_cache["same-model-owner"] = owner
        await self.server._node_cache_lock.acquire()
        try:
            with patch("modiff.NodeBase._server", return_value=self.server), patch(
                "modiff.server.field_action_optional_runtime_requirement", return_value=None,
            ), patch.object(self.server, "queue_message") as publish:
                response = await asyncio.wait_for(self.server.field_action(self.request()), timeout=1)
            self.assertEqual(response.status, 200, response.text)
            self.assertFalse(json.loads(response.text)["error"])
            self.assertGreater(publish.call_count, 3)
            for call in publish.call_args_list:
                self.assertEqual(call.args[0]["workflow_tab_id"], "next-draft")
                self.assertEqual(call.args[0]["sid"], "editing-session")
            self.assertEqual(self.server.node_cache, {"same-model-owner": owner})
            self.assertEqual(owner._sid, "running-session")
            owner.set_filters.assert_not_called()
        finally:
            self.server._node_cache_lock.release()

    def test_only_exact_reviewed_nonqueued_callbacks_are_isolated(self):
        base = {"module": "modules.ModularDiffusers", "action": "ModelsLoader", "fn": "set_filters"}
        self.assertTrue(is_metadata_field_action(base))
        for updates in (
            {"module": "custom.ModularDiffusers"}, {"action": "CustomLoader"},
            {"fn": "execute"}, {"fn": "__del__"}, {"queue": True},
            {"queue": "false"}, {"action": []}, {"fn": {}},
        ):
            with self.subTest(updates=updates):
                self.assertFalse(is_metadata_field_action({**base, **updates}))
        for payload in (None, [], "ModelsLoader"):
            self.assertFalse(is_metadata_field_action(payload))
        with self.assertRaises(ValueError):
            metadata_field_callback("ModelsLoader", "execute", node_id="n", sid="s")

    async def test_metadata_still_requires_authoritative_field_and_optional_runtime(self):
        with patch("modiff.server.metadata_field_callback") as factory:
            response = await self.server.field_action(self.request(fieldKey="undeclared"))
            self.assertEqual(response.status, 400)
            factory.assert_not_called()
            requirement = {
                "schemaVersion": 1, "delivery": "optional_overlay", "requiredNow": True,
                "profileIds": ["missing-profile"], "state": "missing", "reason": "optional_runtime_missing",
            }
            with patch("modiff.server.field_action_optional_runtime_requirement", return_value=requirement):
                response = await self.server.field_action(self.request())
            self.assertEqual(response.status, 409)
            factory.assert_not_called()
            self.server._runtime_mutation_gate = {"token": "activation"}
            response = await self.server.field_action(self.request())
            self.assertEqual(response.status, 409)
            self.assertEqual(json.loads(response.text)["error_code"], "runtime_mutation_busy")
            factory.assert_not_called()

    async def test_queued_metadata_and_custom_callbacks_keep_model_ownership(self):
        for updates in ({"queue": True}, {"module": "custom.Example"}, {"fn": "execute"}):
            with self.subTest(updates=updates):
                await self.server._node_cache_lock.acquire()
                try:
                    with patch.object(self.server, "_field_action", new_callable=AsyncMock) as dispatch:
                        dispatch.return_value = "done"
                        task = asyncio.create_task(self.server.field_action(self.request(**updates)))
                        await asyncio.sleep(0.02)
                        dispatch.assert_not_awaited()
                        self.server._node_cache_lock.release()
                        self.assertEqual(await task, "done")
                finally:
                    if self.server._node_cache_lock.locked():
                        self.server._node_cache_lock.release()

    async def test_cancellation_drains_metadata_before_next_action_or_runtime_activation(self):
        entered, release = threading.Event(), threading.Event()

        def set_filters(values, ref):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release metadata callback")

        with patch("modiff.server.metadata_field_callback", return_value=set_filters), patch(
            "modiff.server.field_action_optional_runtime_requirement", return_value=None,
        ):
            task = asyncio.create_task(self.server.field_action(self.request()))
            following = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                await asyncio.sleep(0.02)
                task.cancel()
                await asyncio.sleep(0.02)
                self.assertFalse(task.done())
                self.assertTrue(self.server._field_metadata_lock.locked())
                self.assertFalse(self.server._node_cache_lock.locked())
                with self.assertRaises(OverlayInstallBusy):
                    self.server._reserve_worker_runtime_gate("activation", "profile")
                mutation = Mock(side_effect=AssertionError("custom code changed during metadata"))
                response = await self.server._extension_mutation(mutation)
                self.assertEqual(response.status, 409)
                mutation.assert_not_called()
                following = asyncio.create_task(self.server.field_action(self.request()))
                await asyncio.sleep(0.02)
                self.assertFalse(following.done())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                if following is not None:
                    self.assertEqual((await following).status, 200)
            token = self.server._reserve_worker_runtime_gate("activation", "profile")
            self.server._release_worker_runtime_gate(token)

    async def test_metadata_waits_until_extension_mutation_finishes(self):
        entered, release = threading.Event(), threading.Event()

        def operation():
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release extension mutation")
            return {"ok": True}

        with patch.object(self.server, "_field_action", new_callable=AsyncMock, return_value="done") as dispatch:
            mutation = asyncio.create_task(self.server._extension_mutation(operation))
            action = None
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                action = asyncio.create_task(self.server.field_action(self.request()))
                await asyncio.sleep(0.02)
                dispatch.assert_not_awaited()
            finally:
                release.set()
                self.assertEqual((await mutation).status, 200)
                if action is not None:
                    self.assertEqual(await action, "done")
                if self.server._model_executor is not None:
                    self.server._model_executor.shutdown()

    def test_context_never_constructs_or_releases_an_executable_node(self):
        from modules.ModularDiffusers import components
        from modules.ModularDiffusers.loaders import ModelsLoader
        from modiff.NodeBase import NodeBase

        with patch.object(ModelsLoader, "__init__", side_effect=AssertionError("model constructor")), patch.object(
            NodeBase, "__init__", side_effect=AssertionError("node constructor"),
        ), patch.object(components, "remove_from_collection") as remove:
            callback = metadata_field_callback("ModelsLoader", "set_filters", node_id="same-model-owner", sid=None)
            callback({"model_type": ""}, {"key": "model_type"})
            self.assertFalse(hasattr(callback.__self__, "execute"))
            self.assertFalse(hasattr(callback.__self__, "__del__"))
            del callback
            remove.assert_not_called()


class ModularMetadataContractTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("transformers"), "requires the reviewed optional runtime")
    def test_all_registered_model_schemas_match_existing_generic_contracts(self):
        from modules.ModularDiffusers.modular_utils import (
            get_all_model_types, pipeline_class_from_model_type, require_modiff_node_contract,
        )
        from modules.ModularDiffusers.field_metadata import _FieldContext

        checked = 0
        for model_type in get_all_model_types():
            if not model_type or model_type == "DummyCustomPipeline":
                continue
            for action, connector in (
                ("EncodePrompt", "text_encoders"), ("Denoise", "unet"),
                ("DecodeLatents", "vae"), ("ImageEncode", "vae"),
                ("ImageEmbeddings", "image_encoder"),
            ):
                with self.subTest(model_type=model_type, action=action):
                    callback = metadata_field_callback(action, "update_node", node_id="draft", sid="editing")
                    try:
                        pipeline = pipeline_class_from_model_type(model_type)
                        _, expected = require_modiff_node_contract(pipeline, callback.__self__.node_type, resolve_blocks=False)
                    except ValueError:
                        with patch.object(_FieldContext, "get_signal_value", return_value=model_type), patch.object(
                            _FieldContext, "send_node_definition",
                        ) as publish, self.assertRaises(ValueError):
                            callback({}, {})
                        publish.assert_called_once_with({})
                        continue
                    expected["params"].pop(connector, None)
                    with patch.object(_FieldContext, "get_signal_value", return_value=model_type), patch.object(
                        _FieldContext, "send_node_definition",
                    ) as publish:
                        callback({}, {})
                    publish.assert_called_once_with(expected["params"])
                    checked += 1
        self.assertGreater(checked, 0)
