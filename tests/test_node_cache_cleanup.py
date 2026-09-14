import asyncio
import json
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from modiff.server import WebServer


class NodeCacheCleanupTests(unittest.IsolatedAsyncioTestCase):
    def server(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return WebServer(modules={}, work_dir=directory.name, data_dir=directory.name)

    def request(self, nodes):
        return SimpleNamespace(json=AsyncMock(return_value={"nodes": nodes}))

    def test_selective_component_destruction_preserves_shared_owners_and_hooks(self):
        server = self.server()
        a, b, shared = object(), object(), object()
        hooks = [SimpleNamespace(model_id=name, hook=SimpleNamespace(other_hooks=[])) for name in ("a", "b", "shared")]
        for hook in hooks:
            hook.hook.other_hooks = [peer for peer in hooks if peer is not hook]
        manager = SimpleNamespace(
            collections={"owner-a": {"a", "shared"}, "owner-b": {"b", "shared"}},
            components={"a": a, "b": b, "shared": shared},
            added_time={"a": 1, "b": 2, "shared": 3}, model_hooks=list(hooks),
            _auto_offload_enabled=True, _auto_offload_device="cuda:0", _offload_strategy="unchanged",
        )
        with patch.dict(sys.modules, {"modules.ModularDiffusers": SimpleNamespace(components=manager)}):
            self.assertEqual(server._release_node_modular_components(["owner-a"]), 1)
            self.assertEqual(manager.collections, {"owner-b": {"b", "shared"}})
            self.assertEqual(manager.components, {"b": b, "shared": shared})
            self.assertEqual(manager.added_time, {"b": 2, "shared": 3})
            self.assertEqual(manager.model_hooks, hooks[1:])
            self.assertTrue(manager._auto_offload_enabled)
            self.assertEqual(manager._offload_strategy, "unchanged")
            self.assertTrue(all(hooks[0] not in hook.hook.other_hooks for hook in manager.model_hooks))
            self.assertEqual(server._release_node_modular_components(["owner-b"]), 2)
            self.assertEqual(manager.components, {})
            self.assertEqual(manager.collections, {})
            self.assertIsNone(manager.model_hooks)
            self.assertFalse(manager._auto_offload_enabled)

    def test_real_diffusers_manager_release_never_moves_discarded_weights_to_cpu(self):
        import torch
        from diffusers.modular_pipelines.components_manager import ComponentsManager, CustomOffloadHook, UserCustomOffloadHook

        server = self.server()
        manager = ComponentsManager()
        doomed, retained = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        doomed_id = manager.add("doomed", doomed, collection="deleted-node")
        retained_id = manager.add("retained", retained, collection="other-node")
        hooks = [UserCustomOffloadHook(key, model, CustomOffloadHook(execution_device="cpu"))
                 for key, model in ((doomed_id, doomed), (retained_id, retained))]
        hooks[0].add_other_hook(hooks[1])
        hooks[1].add_other_hook(hooks[0])
        manager.model_hooks = hooks
        manager._auto_offload_enabled = True
        doomed.to = Mock(side_effect=AssertionError("Destruction must not offload weights"))
        retained.to = Mock(side_effect=AssertionError("Unrelated placement must remain unchanged"))
        manager.enable_auto_cpu_offload = Mock(side_effect=AssertionError("Do not rebuild global hooks"))
        with patch.dict(sys.modules, {"modules.ModularDiffusers": SimpleNamespace(components=manager)}):
            self.assertEqual(server._release_node_modular_components(["deleted-node"]), 1)
        self.assertEqual(set(manager._lookup_ids(collection="deleted-node")), set())
        self.assertEqual(set(manager._lookup_ids(collection="other-node")), {retained_id})
        self.assertIs(manager.components[retained_id], retained)
        self.assertEqual(manager.model_hooks, [hooks[1]])
        self.assertEqual(hooks[1].hook.other_hooks, [])
        doomed.to.assert_not_called()
        retained.to.assert_not_called()

    async def test_slow_destructor_keeps_loop_alive_and_serializes_replacement(self):
        server = self.server()
        entered = threading.Event()
        finish = threading.Event()
        destructor_threads = []

        class SlowNode:
            def __del__(self):
                destructor_threads.append(threading.get_ident())
                entered.set()
                finish.wait(5)

        server.node_cache["loader"] = SlowNode()
        deletion = asyncio.create_task(server.delete_cache(self.request(["loader"])))
        replacement_started = asyncio.Event()

        async def replacement():
            replacement_started.set()
            server.node_cache["loader"] = "replacement"

        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            with patch("modiff.server.import_module", side_effect=AssertionError("No Torch probe during teardown")):
                server._collect_cuda_memory_snapshot()
                server._runtime_accelerator_snapshot(ram_total=128 * 1024**3)
            replacement_task = asyncio.create_task(server._with_node_cache_lease(replacement))
            await asyncio.sleep(0.02)
            self.assertFalse(replacement_started.is_set())
            self.assertNotEqual(destructor_threads, [threading.get_ident()])
            # A disconnected browser must not let a new loader race teardown.
            deletion.cancel()
            await asyncio.sleep(0)
            deletion.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(replacement_started.is_set())
        finally:
            finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await deletion
        await replacement_task
        self.assertEqual(server.node_cache["loader"], "replacement")

    async def test_cache_deletion_waits_for_active_execution_ownership(self):
        server = self.server()
        server.node_cache["active"] = "retained"
        await server._node_cache_lock.acquire()
        deletion = asyncio.create_task(server.delete_cache(self.request("*")))
        await asyncio.sleep(0.02)
        self.assertEqual(server.node_cache["active"], "retained")
        server._node_cache_lock.release()
        response = await deletion
        self.assertEqual(json.loads(response.text)["nodes"], ["active"])
        self.assertEqual(server.node_cache, {})

    async def test_field_actions_wait_for_cache_ownership(self):
        server = self.server()
        server._field_action = AsyncMock(return_value="ok")
        await server._node_cache_lock.acquire()
        action = asyncio.create_task(server.field_action(None))
        await asyncio.sleep(0.02)
        server._field_action.assert_not_awaited()
        server._node_cache_lock.release()
        self.assertEqual(await action, "ok")

    async def test_invalid_node_ids_are_rejected_without_mutation(self):
        server = self.server()
        server.node_cache["keep"] = "retained"
        response = await server.delete_cache(self.request([{}]))
        self.assertEqual(response.status, 400)
        self.assertEqual(server.node_cache, {"keep": "retained"})

    async def test_root_deletion_also_releases_orphans_from_older_effective_graphs(self):
        server = self.server()
        server.node_cache = {"block-v2-node:4:demo:3:old": "orphan", "block-v2-node:5:demo2:3:old": "other"}
        response = await server.delete_cache(self.request(["demo"]))
        self.assertEqual(json.loads(response.text)["nodes"], ["demo", "block-v2-node:4:demo:3:old"])
        self.assertEqual(server.node_cache, {"block-v2-node:5:demo2:3:old": "other"})

    async def test_queued_run_reports_why_it_waits_for_cache_ownership(self):
        server = self.server()
        server.loop = asyncio.get_running_loop()
        waiting = asyncio.Event()
        executed = asyncio.Event()

        def message(payload, *args):
            if payload.get("phase") == "waiting_for_node_cache":
                self.assertIn("node-cache", payload["message"])
                waiting.set()

        server.queue_message = message
        await server._node_cache_lock.acquire()
        owns_test_lock = True
        worker = asyncio.create_task(server._main_worker())
        try:
            await server.queue_task(lambda: server.loop.call_soon_threadsafe(executed.set), (), None, "test", name="Graph execution")
            await asyncio.wait_for(waiting.wait(), 2)
            self.assertFalse(executed.is_set())
            server._node_cache_lock.release()
            owns_test_lock = False
            await asyncio.wait_for(executed.wait(), 2)
        finally:
            if owns_test_lock:
                server._node_cache_lock.release()
            server._shutdown_event.set()
            await asyncio.wait_for(worker, 3)

    async def test_cancel_while_waiting_does_not_start_model_work(self):
        server = self.server()
        server.loop = asyncio.get_running_loop()
        waiting = asyncio.Event()
        terminal = asyncio.Event()
        called = []

        def message(payload, *args):
            if payload.get("phase") == "waiting_for_node_cache":
                waiting.set()
            if payload.get("type") == "task_cancelled":
                terminal.set()

        server.queue_message = message
        server._release_runtime_caches_for_retry = lambda: {}
        await server._node_cache_lock.acquire()
        owns_test_lock = True
        worker = asyncio.create_task(server._main_worker())
        try:
            await server.queue_task(lambda: called.append("model"), (), None, "test", name="Graph execution")
            await asyncio.wait_for(waiting.wait(), 2)
            server.current_task["interrupt_requested"] = True
            server._node_cache_lock.release()
            owns_test_lock = False
            await asyncio.wait_for(terminal.wait(), 2)
            self.assertEqual(called, [])
        finally:
            if owns_test_lock:
                server._node_cache_lock.release()
            server._shutdown_event.set()
            await asyncio.wait_for(worker, 3)
