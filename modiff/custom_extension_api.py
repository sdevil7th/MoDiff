"""HTTP administration for operator-enabled Python, using the existing cache lease."""

import asyncio
import tempfile
from pathlib import Path
import nanoid
from aiohttp import web

from modiff.custom_extensions import ExtensionStore, module_name, python_node_files


class CustomExtensionAPI:
    def _extension_store(self):
        return ExtensionStore()

    def _require_custom_module_enabled(self, module):
        if isinstance(module, str) and module.startswith("custom."):
            return self._extension_store().require_enabled(module.removeprefix("custom."))
        return None

    def _extension_payload(self, *, message=None, removed=()):
        items = self._extension_store().list()
        return {
            "error": False,
            "modules": items,
            "count": len(items),
            "instance": self.instance,
            "message": message,
            "removedCacheNodes": list(removed),
            "root": str(self._extension_store().root),
        }

    async def custom_modules_list(self, request):
        # Listing/refresh are file inspection, not Python reload.
        try:
            return web.json_response(await asyncio.to_thread(self._extension_payload))
        except (ValueError, OSError) as error:
            return web.json_response({"error": True, "message": str(error)}, status=400)

    async def custom_modules_refresh(self, request):
        return await self.custom_modules_list(request)

    async def custom_modules_resolve(self, request):
        # Network metadata inspection stays outside the execution/import lease.
        from modiff.custom_extension_source import resolve_hub_extension

        try:
            body = await self._strict_runtime_control_json(
                request, allowed={"source", "revision"}, required={"source"}
            )
            resolved = await asyncio.to_thread(resolve_hub_extension, **body)
            return web.json_response({"error": False, "source": resolved})
        except (ValueError, OSError) as error:
            return web.json_response({"error": True, "message": str(error)}, status=400)

    async def custom_modules_inspect(self, request):
        try:
            item = await asyncio.to_thread(self._extension_store().inspect, module_name(request.match_info["name"]))
            return web.json_response({"error": False, "module": item})
        except (ValueError, OSError, SyntaxError) as error:
            return web.json_response({"error": True, "message": str(error)}, status=400)

    async def _extension_mutation(self, operation):
        # Do not change imports under either an active graph or a waiting graph.
        if (
            self.current_task or not self.main_queue.empty()
            or self._node_cache_lock.locked() or self._field_metadata_lock.locked()
        ):
            return web.json_response(
                {
                    "error": True,
                    "message": "Wait for the running and queued work and active field updates to finish before changing custom code.",
                },
                status=409,
            )

        async def guarded():
            self._node_cache_teardown_active = True
            try:
                return await self._run_executor_callback(operation)
            finally:
                self._node_cache_teardown_active = False

        try:
            # Import/reload changes the registry used by presentation callbacks.
            # Hold both leases, in this order, until even cancelled threads drain.
            async with self._field_metadata_lock:
                return web.json_response(await self._with_node_cache_lease(guarded))
        except (ValueError, OSError, SyntaxError) as error:
            # Report resulting disabled state too; a failed import is not rollback
            # of Python side effects, and an earlier approval must not survive it.
            return web.json_response({"error": True, "message": str(error)[:2048]}, status=400)

    async def custom_modules_install(self, request):
        try:
            body = await self._strict_runtime_control_json(
                request, allowed={"kind", "source", "name", "revision"}, required={"kind", "source", "name"}
            )
        except ValueError as error:
            return web.json_response({"error": True, "message": str(error)}, status=400)

        def stage():
            item = self._extension_store().stage(**body)
            return {
                **self._extension_payload(message="Source staged with execution disabled. Review before enabling."),
                "module": item,
            }

        return await self._extension_mutation(stage)

    async def custom_modules_add(self, request):
        """The explicit Add action authorizes this exact import, never discovery."""
        try:
            body = await self._strict_runtime_control_json(
                request, allowed={"kind", "source", "name", "revision", "content", "consent"},
                required={"kind", "name", "consent"},
                # A 2 MiB Python file can expand sixfold in JSON \u escapes.
                max_bytes=12 * 1024 * 1024 + 4096,
            )
            if body["consent"] is not True:
                raise ValueError("Adding a node requires permission to run its Python code.")
            name = module_name(body["name"])
            kind = body["kind"]
            if kind not in {"local", "hub", "git", "file"}:
                raise ValueError("Select Local, Hugging Face or Git.")
            if kind == "file":
                content = body.get("content")
                if not isinstance(content, str):
                    raise ValueError("Provide the Python node file contents.")
                files = python_node_files(content.encode("utf-8"))
            else:
                if "content" in body:
                    raise ValueError("File contents are only accepted for a Python file import.")
                source = body.get("source")
                if not isinstance(source, str):
                    raise ValueError("Provide a source path or repository.")
                if kind in {"git", "hub"}:
                    from modiff.custom_extension_source import resolve_git_extension, resolve_hub_extension

                    resolver = resolve_git_extension if kind == "git" else resolve_hub_extension
                    resolved = await asyncio.to_thread(resolver, source, body.get("revision"))
                    source, body["revision"] = resolved["source"], resolved["revision"]
        except (ValueError, OSError, SyntaxError) as error:
            return web.json_response({"error": True, "message": str(error)}, status=400)

        def add():
            from modules import MODULE_MAP

            store = self._extension_store()
            if kind == "file":
                with tempfile.TemporaryDirectory(prefix="modiff-node-upload-") as temporary:
                    for filename, data in files.items():
                        (Path(temporary) / filename).write_bytes(data)
                    item = store.stage(kind="local", source=temporary, name=name)
            else:
                item = store.stage(kind=kind, source=source, name=name, revision=body.get("revision"))
            registry = store.enable(name, code_hash=item["codeHash"], consent=True)
            self.modules[item["moduleKey"]] = registry
            MODULE_MAP[item["moduleKey"]] = registry
            self.instance = nanoid.generate(size=10)
            return {**self._extension_payload(message="Custom nodes added and enabled."), "module": store.inspect(name)}

        return await self._extension_mutation(add)

    async def custom_modules_enable(self, request):
        try:
            name = module_name(request.match_info["name"])
            body = await self._strict_runtime_control_json(
                request, allowed={"codeHash", "consent"}, required={"codeHash", "consent"}
            )
            # Validate consent and preview identity before clearing any caches.
            store = self._extension_store()
            item = await asyncio.to_thread(store.inspect, name)
            if body["consent"] is not True or body["codeHash"] != item["codeHash"]:
                raise ValueError("Explicit consent for the current code hash is required. Inspect the source again.")
        except (ValueError, OSError, SyntaxError) as error:
            return web.json_response({"error": True, "message": str(error)}, status=400)

        def enable():
            from modules import MODULE_MAP

            key = f"custom.{name}"
            # Check again under the execution lease, before invalidating results.
            if store.inspect(name)["codeHash"] != body["codeHash"]:
                raise ValueError("Source changed after inspection; review again.")
            removed = self._release_cached_nodes(
                [node_id for node_id, node in self.node_cache.items() if getattr(node, "module_name", None) == key]
            )
            self.modules.pop(key, None)
            MODULE_MAP.pop(key, None)
            registry = store.enable(name, code_hash=body["codeHash"], consent=body["consent"])
            self.modules[key] = registry
            MODULE_MAP[key] = registry
            self.instance = nanoid.generate(size=10)
            return self._extension_payload(
                message="Custom code enabled. Its cached dependents were released.", removed=removed
            )

        return await self._extension_mutation(enable)

    async def custom_modules_reload(self, request):
        return await self.custom_modules_enable(request)

    async def custom_modules_disable(self, request):
        try:
            name = module_name(request.match_info["name"])
            await self._strict_runtime_control_json(request, allowed=set(), allow_empty=True)
        except ValueError as error:
            return web.json_response({"error": True, "message": str(error)}, status=400)

        def disable():
            from modules import MODULE_MAP

            key = f"custom.{name}"
            removed = self._release_cached_nodes(
                [node_id for node_id, node in self.node_cache.items() if getattr(node, "module_name", None) == key]
            )
            self._extension_store().disable(name)
            self.modules.pop(key, None)
            MODULE_MAP.pop(key, None)
            self.instance = nanoid.generate(size=10)
            return self._extension_payload(
                message="Custom code disabled. Restart to remove any import side effects.", removed=removed
            )

        return await self._extension_mutation(disable)

    async def custom_modules_update(self, request):
        return web.json_response(
            {
                "error": True,
                "message": "Moving-branch updates are retired. Stage an exact revision under a new module name, inspect it, then enable explicitly.",
            },
            status=409,
        )
