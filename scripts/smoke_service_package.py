"""No-model HTTP smoke in temporary storage; suitable for clean CPU CI installs."""

import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def smoke():
    # Redirect every path before importing the singleton/queue reconciliation.
    with tempfile.TemporaryDirectory(prefix="modiff-service-smoke-") as temporary, ExitStack() as isolation:
        from modiff.config import CONFIG

        for key in CONFIG.paths:
            if key != "app_root":
                directory = Path(temporary) / key
                directory.mkdir(parents=True, exist_ok=True)
                CONFIG.paths[key] = str(directory)
        CONFIG.server["host"] = "127.0.0.1"
        CONFIG.server["port"] = 0
        os.environ["HF_HUB_OFFLINE"] = "1"
        from modiff.custom_extensions import ExtensionStore

        original_init = ExtensionStore.__init__

        def isolated_extension_store(self, root=None):
            original_init(self, root if root is not None else Path(temporary) / "custom")

        # Registry construction occurs during server import, before an API
        # store can be replaced. Never discover the operator's extensions here.
        isolation.enter_context(patch.object(ExtensionStore, "__init__", isolated_extension_store))
        from modiff.server import server
        from modiff.service import request, run
        from aiohttp import ClientSession

        store = ExtensionStore(Path(temporary) / "custom")
        server._extension_store = lambda: store
        await server.run()
        origin = f"http://127.0.0.1:{server.site._server.sockets[0].getsockname()[1]}"

        async def http(path, body=None):
            return await asyncio.to_thread(request, origin, path, body)

        try:
            health = await http("/health")
            assert health.get("error") is not True
            async with ClientSession() as client:
                for path, expected_type in (
                    ("/", "text/html"),
                    ("/assets/index.js", "javascript"),
                    ("/favicon.ico", "image/"),
                ):
                    async with client.get(origin + path) as response:
                        assert response.status == 200, (path, response.status)
                        assert expected_type in response.headers.get("Content-Type", ""), path
                        assert await response.content.read(128), f"Empty bundled asset: {path}"
            graph = json.loads((ROOT / "examples/service/text-api.json").read_text())
            interface = json.loads((ROOT / "examples/service/interface.json").read_text())
            values = json.loads((ROOT / "examples/service/inputs.json").read_text())
            direct = await http("/graph", graph)
            for _ in range(240):
                receipt = await http(f"/runs/{direct['task_id']}")
                if receipt["task"].get("status") in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.25)
            assert receipt["task"]["status"] == "completed", receipt["task"]
            candidates = await http("/service_package", {"operation": "inspect", "graph": graph})
            assert any(item["nodeId"] == "prompt" for item in candidates["inputs"])
            package = (await http("/service_package", {"operation": "build", "graph": graph, "interface": interface}))[
                "package"
            ]
            from modiff.service_package import service_outputs

            baseline = service_outputs(package, receipt, direct["task_id"])["outputs"]
            for mode in ("manual", "auto"):
                graph["runtimeHints"] = {"resourceMode": mode}
                package = (
                    await http("/service_package", {"operation": "build", "graph": graph, "interface": interface})
                )["package"]
                result = await asyncio.to_thread(run, origin, package, values, timeout=60)
                assert result["outputs"]["text"][0]["value"] == baseline["text"][0]["value"]
            staged = await http(
                "/custom_modules/install",
                {
                    "kind": "local",
                    "source": str(ROOT / "examples/custom_nodes/PromptTools"),
                    "name": "ServiceExample",
                },
            )
            await http(
                "/custom_modules/ServiceExample/enable", {"codeHash": staged["module"]["codeHash"], "consent": True}
            )
            graph["nodes"]["prefix"] = {
                "module": "custom.ServiceExample",
                "action": "PromptPrefix",
                "params": {"text": {"sourceId": "prompt", "sourceKey": "output"}, "prefix": {"value": "Service:"}},
            }
            graph["nodes"]["preview"]["params"]["value"] = {"sourceId": "prefix", "sourceKey": "result"}
            graph["paths"] = [["prompt", "prefix", "preview"]]
            graph["runtimeHints"] = {"resourceMode": "manual"}
            package = (await http("/service_package", {"operation": "build", "graph": graph, "interface": interface}))[
                "package"
            ]
            assert package["requirements"]["customNodes"][0]["codeHash"] == staged["module"]["codeHash"]
            result = await asyncio.to_thread(run, origin, package, values, timeout=60)
            assert result["outputs"]["text"][0]["value"] == ["Service: " + values["prompt"]]
            source = store.path("ServiceExample") / "main.py"
            source.write_text(source.read_text() + "\n# source changed\n")
            try:
                await asyncio.to_thread(run, origin, package, values, timeout=60)
            except ValueError as error:
                assert "changed" in str(error)
            else:
                raise AssertionError("Changed custom code was accepted.")
            print(
                json.dumps(
                    {
                        "health": "passed",
                        "bundledAssets": "passed",
                        "savedGraph": "passed",
                        "serviceManual": "passed",
                        "serviceAuto": "passed",
                        "customServiceAndDrift": "passed",
                        "modelDownloads": 0,
                    }
                )
            )
        finally:
            queue = await http("/queue")
            assert queue["current"] is None and not queue["queued"], "Owned smoke queue did not drain."
            await server.cleanup()


if __name__ == "__main__":
    asyncio.run(smoke())
