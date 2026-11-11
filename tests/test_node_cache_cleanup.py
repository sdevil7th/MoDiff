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

    def cache_status_handler(self, server):
        routes = [route for route in server.app.router.routes()
                  if route.method == "GET" and route.resource.canonical == "/cache"]
        self.assertEqual(len(routes), 1, "GET /cache must expose the current worker's cache roster")
        return routes[0].handler

    def test_cache_status_get_route_preserves_cache_delete_route(self):
        server = self.server()
        get_handler = self.cache_status_handler(server)
        delete_routes = [route for route in server.app.router.routes()
                         if route.method == "DELETE" and route.resource.canonical == "/cache"]
        self.assertEqual(len(delete_routes), 1)
        self.assertEqual(delete_routes[0].handler, server.delete_cache)
        self.assertNotEqual(get_handler, delete_routes[0].handler)

    async def test_cache_status_returns_exact_roster_without_inspecting_cached_objects(self):
        server = self.server()

        class OpaqueCachedNode:
            def __init__(self):
                raise AssertionError("Cache status must not construct cached nodes")

            def __getattribute__(self, name):
                raise AssertionError(f"Cache status must not inspect cached node attribute {name}")

            def __bool__(self):
                raise AssertionError("Cache status must not evaluate cached nodes")

            def __call__(self, *args, **kwargs):
                raise AssertionError("Cache status must not construct or execute nodes")

            def __str__(self):
                raise AssertionError("Cache status must not serialize cached nodes")

            def __repr__(self):
                raise AssertionError("Cache status must not represent cached nodes")

        expected_nodes = ["loader", "block-v2-node:5:block:6:encode", "preview"]
        server.node_cache = {node_id: object.__new__(OpaqueCachedNode) for node_id in expected_nodes}
        request = SimpleNamespace(json=AsyncMock(side_effect=AssertionError("GET must not read a body")))
        forbidden = AssertionError("Cache status must not acquire model ownership or release cache")
        with (
            patch.object(server, "_with_node_cache_lease", AsyncMock(side_effect=forbidden)),
            patch.object(server, "_run_executor_callback", AsyncMock(side_effect=forbidden)),
            patch.object(server, "_release_cached_nodes", side_effect=forbidden),
            patch("modiff.server.import_module", side_effect=forbidden),
        ):
            response = await self.cache_status_handler(server)(request)

        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.text), {
            "error": False, "instance": server.instance, "cachedNodes": expected_nodes,
        })
        self.assertEqual(list(server.node_cache), expected_nodes)
        request.json.assert_not_awaited()

    async def test_cache_status_does_not_wait_for_active_cache_ownership(self):
        server = self.server()
        server.node_cache["active-loader"] = object()
        server.current_task = {"task_id": "active-service-task", "status": "running"}
        handler = self.cache_status_handler(server)
        async with server._node_cache_lock:
            response = await asyncio.wait_for(handler(SimpleNamespace()), timeout=0.25)
            self.assertTrue(server._node_cache_lock.locked())
        self.assertEqual(json.loads(response.text), {
            "error": False, "instance": server.instance, "cachedNodes": ["active-loader"],
        })

    async def test_cache_status_response_is_independent_of_later_cache_mutation(self):
        server = self.server()
        server.node_cache = {"first": object(), "retained": object()}
        handler = self.cache_status_handler(server)
        response = await handler(SimpleNamespace())
        server.node_cache.pop("first")
        server.node_cache["new"] = object()
        self.assertEqual(json.loads(response.text), {
            "error": False, "instance": server.instance, "cachedNodes": ["first", "retained"],
        })
        current = await handler(SimpleNamespace())
        self.assertEqual(json.loads(current.text)["cachedNodes"], ["retained", "new"])

    async def test_cache_status_empty_roster_reports_its_own_worker_instance(self):
        for instance in ("original-worker", "replacement-worker"):
            with self.subTest(instance=instance):
                server = self.server()
                server.instance = instance
                response = await self.cache_status_handler(server)(SimpleNamespace())
                self.assertEqual(json.loads(response.text), {
                    "error": False, "instance": instance, "cachedNodes": [],
                })

    async def test_failed_worker_releases_exception_tensor_references_before_next_task(self):
        import gc
        import weakref
        server = self.server()
        server.loop = asyncio.get_running_loop()
        refs, observations = [], []
        class Model: pass
        def allocate_then_fail():
            model = Model()
            refs.append(weakref.ref(model))
            try:
                raise RuntimeError('HIP out of memory')
            except RuntimeError as cause:
                raise RuntimeError('Error executing model') from cause
        def retry():
            gc.collect()
            observations.append(refs[0]() is None)
        future = server.loop.create_future()
        await server.queue_task(allocate_then_fail, (), future, 'test', name='Failure probe')
        await server.queue_task(retry, (), None, 'test', name='Retry probe')
        server.main_queue.put_nowait(None)
        try:
            with patch.object(server, '_best_effort_device_cache_clear', return_value=[]):
                await asyncio.wait_for(server._main_worker(), timeout=10)
            self.assertEqual(observations, [True])
            error = future.exception()
            self.assertEqual(server._classify_exception(error)['category'], 'oom')
            self.assertEqual(server.recent_tasks[0]['status'], 'completed')
            self.assertEqual(server.recent_tasks[1]['status'], 'failed')
        finally:
            if future.done():
                future.exception()
            await server.cleanup()

    async def test_release_drops_transitive_consumers_but_preserves_unrelated_shared_owner(self):
        server = self.server()
        node = lambda *sources: SimpleNamespace(_cache_input_sources=frozenset(sources), _mm_models=[])
        server.node_cache = {'load': node(), 'encode': node('load'), 'denoise': node('encode'),
                             'other-load': node(), 'other-output': node('other-load')}
        shared = object()
        manager = SimpleNamespace(collections={'load': {'weights'}, 'other-load': {'weights'}},
                                  components={'weights': shared})
        with patch.dict(sys.modules, {'modules.ModularDiffusers': SimpleNamespace(components=manager)}):
            result = json.loads((await server.delete_cache(self.request(['load']))).text)
        self.assertEqual(set(result['nodes']), {'load', 'encode', 'denoise'})
        self.assertEqual(set(server.node_cache), {'other-load', 'other-output'})
        self.assertIs(manager.components['weights'], shared)

    async def test_standard_release_preserves_shared_models_without_cpu_offload(self):
        server = self.server()
        removed, shared = object(), object()
        class Owner:
            def __init__(self, ids):
                self._mm_models = ids
            def __del__(self):
                for key in self._mm_models:
                    manager.remove(key)
        manager = SimpleNamespace(cache={'removed': removed, 'shared': shared}, remove=Mock())
        server.node_cache = {'remove': Owner(['removed', 'shared']), 'keep': Owner(['shared'])}
        with patch('modiff.server.memory_manager', manager):
            await server.delete_cache(self.request(['remove']))
        self.assertEqual(manager.cache, {'shared': shared})
        self.assertEqual(server.node_cache['keep']._mm_models, ['shared'])
        manager.remove.assert_not_called()
        server.node_cache['keep']._mm_models = []

    async def test_model_callbacks_keep_one_worker_while_http_pool_remains_available(self):
        server = self.server()
        try:
            ids = [await server._run_executor_callback(threading.get_ident) for _ in range(5)]
            self.assertEqual(len(set(ids)), 1)
            started, release = threading.Event(), threading.Event()
            def block():
                started.set()
                release.wait(2)
            task = asyncio.create_task(server._run_executor_callback(block))
            while not started.is_set():
                await asyncio.sleep(0.001)
            try:
                control_id = await asyncio.wait_for(
                    server._run_executor_callback(threading.get_ident, model_work=False), timeout=0.5,
                )
                self.assertNotEqual(control_id, ids[0])
            finally:
                release.set()
                await task
        finally:
            await server.cleanup()

    async def test_output_invalidation_retains_component_owners_and_objects(self):
        server = self.server()
        loader = SimpleNamespace(_mm_models=[], invalidate_cache=Mock())
        standard = SimpleNamespace(_mm_models=['pipeline'], invalidate_cache=Mock())
        encode = SimpleNamespace(_mm_models=[], invalidate_cache=Mock())
        nested = 'block-v2-node:5:block:6:encode'
        server.node_cache = {'load': loader, 'standard': standard, nested: encode}
        manager = SimpleNamespace(collections={'load': {'weights'}}, components={'weights': object()})
        before = dict(server.node_cache)
        request = SimpleNamespace(json=AsyncMock(return_value={'nodes': ['load', 'standard', 'block'], 'scope': 'outputs'}))
        with patch.dict(sys.modules, {'modules.ModularDiffusers': SimpleNamespace(components=manager)}):
            response = await server.delete_cache(request)
        result = json.loads(response.text)
        self.assertEqual(result['nodes'], [nested])
        self.assertEqual(result['retainedModelNodes'], ['load', 'standard'])
        self.assertEqual(server.node_cache, before)
        self.assertEqual(set(manager.components), {'weights'})
        encode.invalidate_cache.assert_called_once_with()
        loader.invalidate_cache.assert_not_called()
        standard.invalidate_cache.assert_not_called()

    async def test_output_invalidation_waits_for_active_execution(self):
        server = self.server()
        node = SimpleNamespace(_mm_models=[], invalidate_cache=Mock())
        server.node_cache['encode'] = node
        await server._node_cache_lock.acquire()
        request = SimpleNamespace(json=AsyncMock(return_value={'nodes': ['encode'], 'scope': 'outputs'}))
        action = asyncio.create_task(server.delete_cache(request))
        await asyncio.sleep(0.02)
        node.invalidate_cache.assert_not_called()
        server._node_cache_lock.release()
        self.assertEqual((await action).status, 200)
        node.invalidate_cache.assert_called_once_with()
        self.assertIs(server.node_cache['encode'], node)

    async def test_unknown_cache_scope_cannot_destroy_model_objects(self):
        server = self.server()
        server.node_cache['load'] = object()
        request = SimpleNamespace(json=AsyncMock(return_value={'nodes': '*', 'scope': 'typo'}))
        self.assertEqual((await server.delete_cache(request)).status, 400)
        self.assertIn('load', server.node_cache)

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
        action = asyncio.create_task(server.field_action(self.request([])))
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
