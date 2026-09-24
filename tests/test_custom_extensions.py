"""Extension discovery must not import code or silently grant execution authority."""

import json
import sys
from pathlib import Path

import pytest

from modiff.custom_extensions import ExtensionStore, ExtensionError


def source_node(root, *, expression="text.upper()"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("from .main import Echo\n")
    (root / "main.py").write_text(
        "from modiff.NodeBase import NodeBase\n"
        "class Echo(NodeBase):\n"
        '    label = "Custom Echo"\n'
        '    category = "Text"\n'
        '    params = {"text": {"type": "string", "default": "hello"}, '
        '"out": {"type": "string", "display": "output"}}\n'
        f'    def execute(self, text): return {{"out": {expression}}}\n'
    )
    return root


@pytest.fixture
def store(tmp_path):
    store = ExtensionStore(tmp_path / "custom")
    yield store
    store.unload("Example")


def test_staging_discovery_and_declining_review_never_import(store, tmp_path):
    source = source_node(tmp_path / "source")
    marker = tmp_path / "executed"
    with (source / "__init__.py").open("a") as f:
        f.write(f'open({str(marker)!r}, "w").write("executed")\n')
    item = store.stage(kind="local", source=str(source), name="Example")
    assert item["status"] == "disabled"
    assert item["preview"]["nodes"]["Echo"]["params"]["out"]["type"] == "string"
    assert store.list()[0]["codeHash"] == item["codeHash"]
    assert not marker.exists()
    with pytest.raises(ExtensionError, match="consent"):
        store.enable("Example", code_hash=item["codeHash"], consent=False)
    assert not marker.exists()


def test_exact_code_enable_reload_and_change_rejection(store, tmp_path, monkeypatch):
    item = store.stage(kind="local", source=str(source_node(tmp_path / "source")), name="Example")
    registry = store.enable("Example", code_hash=item["codeHash"], consent=True)
    from modules import MODULE_MAP

    monkeypatch.setitem(MODULE_MAP, "custom.Example", registry)
    assert registry["Echo"]["params"]["out"]["type"] == "string"
    action = sys.modules["custom.Example.main"].Echo
    assert action("test").execute("Hello") == {"out": "HELLO"}
    path = Path(store.inspect("Example")["path"]) / "main.py"
    path.write_text(path.read_text().replace("text.upper()", "text.lower()"))
    with pytest.raises(ExtensionError, match="changed"):
        store.require_enabled("Example")
    with pytest.raises(ExtensionError, match="changed"):
        store.enable("Example", code_hash=item["codeHash"], consent=True)
    changed = store.inspect("Example")
    store.enable("Example", code_hash=changed["codeHash"], consent=True)
    assert sys.modules["custom.Example.main"].Echo("test").execute("Hello") == {"out": "hello"}
    assert sys.modules["custom.Example.main"].Echo is not action


def test_path_escape_and_missing_dependencies_do_not_execute(store, tmp_path):
    source = source_node(tmp_path / "source")
    (source / "escape.py").symlink_to(tmp_path / "private.py")
    with pytest.raises(ExtensionError, match="link"):
        store.stage(kind="local", source=str(source), name="Example")
    (source / "escape.py").unlink()
    (source / "requirements.txt").write_text("modiff-nonexistent-test-package==1.0\n")
    item = store.stage(kind="local", source=str(source), name="Example")
    assert item["dependencies"][0]["status"] == "missing"
    with pytest.raises(ExtensionError, match="dependencies"):
        store.enable("Example", code_hash=item["codeHash"], consent=True)
    assert "custom.Example.main" not in sys.modules


def test_import_failure_is_recorded_and_does_not_enable(store, tmp_path):
    source = source_node(tmp_path / "source")
    (source / "__init__.py").write_text('raise RuntimeError("broken import")\n')
    item = store.stage(kind="local", source=str(source), name="Example")
    with pytest.raises(ExtensionError, match="broken import"):
        store.enable("Example", code_hash=item["codeHash"], consent=True)
    failed = store.inspect("Example")
    assert not failed["enabled"]
    assert "broken import" in failed["diagnostic"]
    assert "custom.Example" not in sys.modules


def test_git_requires_an_immutable_revision_before_clone(store):
    with pytest.raises(ExtensionError, match="40-character"):
        store.stage(kind="git", source="https://example.com/example.git", name="Example", revision="main")


def test_unapproved_directory_is_not_loaded_at_startup(store, tmp_path):
    source_node(store.root / "Example")
    registry = {}
    store.load_enabled(registry)
    assert registry == {}
    assert "custom.Example.main" not in sys.modules
    assert store.inspect("Example")["status"] == "disabled"


def test_approval_is_outside_source_and_hash_includes_helper_and_dependencies(store, tmp_path):
    source = source_node(tmp_path / "source")
    (source / "helper.py").write_text("VALUE = 1\n")
    (source / ".extensions.json").write_text(json.dumps({"Example": {"enabled": True}}))
    item = store.stage(kind="local", source=str(source), name="Example")
    assert not item["enabled"]
    helper = Path(item["path"]) / "helper.py"
    helper.write_text("VALUE = 2\n")
    assert store.inspect("Example")["codeHash"] != item["codeHash"]


def test_pinned_hub_modular_block_executes_through_native_upstream_and_reloads(store, tmp_path, monkeypatch):
    import shutil
    from modules import MODULE_MAP

    fixture = Path(__file__).resolve().parents[1] / "examples/custom_nodes/ModularPrompt"
    revision = "a" * 40
    snapshot = tmp_path / "models--fixture--prompt" / "snapshots" / revision
    shutil.copytree(fixture, snapshot)
    calls = []

    def downloaded(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", downloaded)
    item = store.stage(kind="hub", source="fixture/prompt", name="Example", revision=revision)
    assert calls[0][1]["revision"] == revision
    assert "*.safetensors" not in calls[0][1]["allow_patterns"]
    assert item["preview"]["nodes"]["Block"]["params"]["out_result"]["type"] == "string"
    registry = store.enable("Example", code_hash=item["codeHash"], consent=True)
    monkeypatch.setitem(MODULE_MAP, "custom.Example", registry)
    klass = sys.modules["custom.Example.main"].Block
    assert klass("custom-block").execute(text="test") == {"out_result": "test — modular"}
    code = Path(item["path"]) / "block.py"
    code.write_text(code.read_text(encoding="utf-8").replace(" — modular", " — changed"), encoding="utf-8")
    changed = store.inspect("Example")
    store.enable("Example", code_hash=changed["codeHash"], consent=True)
    assert sys.modules["custom.Example.main"].Block("custom-block").execute(text="test") == {
        "out_result": "test — changed"
    }


def test_mellon_omitted_model_inputs_are_reviewable_without_changing_approved_bytes(store, tmp_path, monkeypatch):
    import shutil
    from modules import MODULE_MAP
    from modules.ModularDiffusers.pipeline_schema import MoDiffPipelineConfig

    source = tmp_path / "mellon-source"
    shutil.copytree(Path(__file__).resolve().parents[1] / "examples/custom_nodes/ModularPrompt", source)
    sidecar = source / "mellon_pipeline_config.json"
    metadata = json.loads(sidecar.read_bytes())
    del metadata["node_params"]["custom"]["model_input_names"]
    raw = json.dumps(metadata).encode()
    sidecar.write_bytes(raw)
    # The historical declarative pipeline loader retains its strict contract.
    with pytest.raises(OSError, match="model_input_names"):
        MoDiffPipelineConfig.from_json_bytes(raw)
    item = store.stage(kind="local", source=str(source), name="Example")
    assert item["preview"]["contract"]["model_input_names"] == []
    assert (Path(item["path"]) / sidecar.name).read_bytes() == raw
    assert not item["enabled"] and "custom.Example.main" not in sys.modules
    registry = store.enable("Example", code_hash=item["codeHash"], consent=True)
    monkeypatch.setitem(MODULE_MAP, "custom.Example", registry)
    assert sys.modules["custom.Example.main"].Block("mellon").execute(text="test") == {"out_result": "test — modular"}


@pytest.mark.parametrize("value", [None, "", {}, [None], [""]])
def test_mellon_model_input_defaults_do_not_accept_invalid_declared_values(store, tmp_path, value):
    import shutil

    source = tmp_path / "mellon-source"
    shutil.copytree(Path(__file__).resolve().parents[1] / "examples/custom_nodes/ModularPrompt", source)
    sidecar = source / "mellon_pipeline_config.json"
    metadata = json.loads(sidecar.read_bytes())
    metadata["node_params"]["custom"]["model_input_names"] = value
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(OSError, match="model_input_names"):
        store.stage(kind="local", source=str(source), name="Example")
    assert not store.path("Example").exists()


@pytest.fixture
def backend(store, tmp_path, monkeypatch):
    from modiff.config import CONFIG

    for key in CONFIG.paths:
        if key != "app_root":
            directory = tmp_path / "runtime" / key
            directory.mkdir(parents=True, exist_ok=True)
            monkeypatch.setitem(CONFIG.paths, key, str(directory))
    from modiff.server import WebServer
    from modules import MODULE_MAP

    server = WebServer(modules=dict(MODULE_MAP), work_dir=str(tmp_path), data_dir=str(tmp_path / "data"))
    monkeypatch.setattr(server, "_extension_store", lambda: store)
    monkeypatch.setattr("modiff.NodeBase._server", lambda: server)
    yield server
    MODULE_MAP.pop("custom.Example", None)
    if server._model_executor:
        server._model_executor.shutdown(wait=True)


def request(body=None, name="Example"):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    return SimpleNamespace(json=AsyncMock(return_value=body or {}), match_info={"name": name})


def test_api_reload_invalidates_only_affected_nodes_and_rejects_busy_or_unapproved(backend, store, tmp_path):
    import asyncio
    from types import SimpleNamespace

    async def scenario():
        installed = await backend.custom_modules_install(
            request({"kind": "local", "source": str(source_node(tmp_path / "source")), "name": "Example"})
        )
        item = json.loads(installed.text)["module"]
        assert "custom.Example" not in backend.modules
        denied = await backend.custom_modules_enable(request({"codeHash": item["codeHash"], "consent": False}))
        assert denied.status == 400
        approved = await backend.custom_modules_enable(request({"codeHash": item["codeHash"], "consent": True}))
        assert approved.status == 200, approved.text
        node = {"module": "custom.Example", "action": "Echo", "params": {"text": {"value": "Hello"}}}
        backend.execute_node("echo", node, "test", quiet=True)
        assert backend.node_cache["echo"].output == {"out": "HELLO"}
        backend.execute_node("echo", node, "test", quiet=True)
        assert not backend.node_cache["echo"]._has_changed
        backend.node_cache["dependent"] = SimpleNamespace(_cache_input_sources={"echo"})
        backend.node_cache["unrelated"] = SimpleNamespace(_cache_input_sources=set())
        code = Path(item["path"]) / "main.py"
        code.write_text(code.read_text().replace("text.upper()", "text.lower()"))
        with pytest.raises(ExtensionError, match="changed"):
            backend.execute_node("echo", node, "test", quiet=True)
        changed = store.inspect("Example")
        backend.current_task = {"task_id": "running"}
        busy = await backend.custom_modules_reload(request({"codeHash": changed["codeHash"], "consent": True}))
        assert busy.status == 409
        assert "echo" in backend.node_cache
        backend.current_task = None
        refreshed = await backend.custom_modules_refresh(request())
        assert refreshed.status == 200
        assert "echo" in backend.node_cache
        reloaded = await backend.custom_modules_reload(request({"codeHash": changed["codeHash"], "consent": True}))
        assert set(json.loads(reloaded.text)["removedCacheNodes"]) == {"echo", "dependent"}
        assert set(backend.node_cache) == {"unrelated"}
        backend.execute_node("echo", node, "test", quiet=True)
        assert backend.node_cache["echo"].output == {"out": "hello"}

    asyncio.run(scenario())


def test_documented_prompt_tools_example_stages_and_executes_through_the_graph(backend, store, monkeypatch):
    """Keep the public walkthrough tied to the checked-in executable example."""
    from modiff.config import CONFIG
    from modules import MODULE_MAP

    source = Path(__file__).resolve().parents[1] / "examples/custom_nodes/PromptTools"
    item = store.stage(kind="local", source=str(source), name="PromptTools")
    assert not item["enabled"]
    definition = item["preview"]["nodes"]["PromptPrefix"]
    assert definition["params"]["text"]["display"] == "textarea"
    assert definition["params"]["prefix"]["display"] == "textarea"
    assert definition["params"]["prompt_input"]["display"] == "input"
    assert definition["params"]["prompt_input"]["required"] is False
    assert definition["params"]["result"]["display"] == "output"

    registry = store.enable("PromptTools", code_hash=item["codeHash"], consent=True)
    assert registry["PromptPrefix"]["resizable"] is True
    monkeypatch.setitem(MODULE_MAP, "custom.PromptTools", registry)
    backend.modules["custom.PromptTools"] = registry
    try:
        backend.execute_node(
            "documented-text-value",
            {
                "module": "modules.Primitive",
                "action": "TextValue",
                "params": {"text": {"value": "a lighthouse at night"}},
            },
            "test",
            quiet=True,
        )
        prefix_node = {
            "module": "custom.PromptTools",
            "action": "PromptPrefix",
            "params": {
                "text": {"value": "this inline value is replaced by the connected prompt"},
                "prompt_input": {"sourceId": "documented-text-value", "sourceKey": "output"},
                "prefix": {"value": "Watercolor:"},
            },
        }
        backend.execute_node("documented-prompt-prefix", prefix_node, "test", quiet=True)
        assert backend.node_cache["documented-prompt-prefix"].output == {
            "result": "Watercolor: a lighthouse at night"
        }
        destination = Path(CONFIG.paths["data"]) / "exports" / "PromptPrefix_test.txt"
        backend.execute_node(
            "documented-export",
            {
                "module": "modules.Primitive",
                "action": "ExportData",
                "params": {
                    "value": {"sourceId": "documented-prompt-prefix", "sourceKey": "result"},
                    "filename": {"value": str(destination)},
                    "format": {"value": "text"},
                },
            },
            "test",
            quiet=True,
        )
        assert destination.read_text(encoding="utf-8") == "Watercolor: a lighthouse at night\n"
        assert backend.node_cache["documented-export"].output["output"] == "Watercolor: a lighthouse at night"
    finally:
        backend.modules.pop("custom.PromptTools", None)
        store.unload("PromptTools")


def test_reload_client_cancellation_keeps_the_existing_execution_lease(backend, store, tmp_path, monkeypatch):
    import asyncio
    import threading

    item = store.stage(kind="local", source=str(source_node(tmp_path / "source")), name="Example")
    entered, release = threading.Event(), threading.Event()
    original = store.enable

    def pause(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "enable", pause)

    async def scenario():
        task = asyncio.create_task(
            backend.custom_modules_enable(request({"codeHash": item["codeHash"], "consent": True}))
        )
        try:
            while not entered.is_set():
                await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            assert backend._node_cache_lock.locked()
            assert not task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not backend._node_cache_lock.locked()
        assert store.inspect("Example")["enabled"]

    asyncio.run(scenario())


@pytest.mark.parametrize("busy", ["running", "queued", "lease"])
def test_source_resolution_and_inspection_stay_available_during_execution(backend, store, tmp_path, monkeypatch, busy):
    import asyncio
    import threading

    item = store.stage(kind="local", source=str(source_node(tmp_path / "source")), name="Example")
    entered, release = threading.Event(), threading.Event()
    identity = {"kind": "hub", "source": "example/block", "requestedRevision": "main", "revision": "a" * 40}

    def resolve(**kwargs):
        assert kwargs == {"source": "example/block"}
        entered.set()
        assert release.wait(5)
        return identity

    monkeypatch.setattr("modiff.custom_extension_source.resolve_hub_extension", resolve)

    async def scenario():
        if busy == "running":
            backend.current_task = {"task_id": "running"}
        elif busy == "queued":
            await backend.main_queue.put({"task_id": "waiting"})
        else:
            await backend._node_cache_lock.acquire()
        lookup = asyncio.create_task(backend.custom_modules_resolve(request({"source": "example/block"})))
        try:
            async with asyncio.timeout(3):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
                inspected = await backend.custom_modules_inspect(request())
                assert json.loads(inspected.text)["module"]["codeHash"] == item["codeHash"]
                listed = await backend.custom_modules_list(request())
                assert listed.status == 200
                denied = await backend.custom_modules_enable(request({"codeHash": item["codeHash"], "consent": True}))
                assert denied.status == 409
                assert "running and queued work" in json.loads(denied.text)["message"]
                assert "custom.Example.main" not in sys.modules
        finally:
            release.set()
            response = await lookup
            backend.current_task = None
            if busy == "queued":
                backend.main_queue.get_nowait()
            if busy == "lease":
                backend._node_cache_lock.release()
        assert response.status == 200
        assert json.loads(response.text)["source"] == identity
        assert not store.inspect("Example")["enabled"]

    asyncio.run(scenario())


def test_source_resolution_cancellation_does_not_stage_or_acquire_import_lease(backend, store, monkeypatch):
    import asyncio
    import threading

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def resolve(**kwargs):
        entered.set()
        assert release.wait(5)
        finished.set()
        return {"kind": "hub", "source": "example/block", "requestedRevision": "main", "revision": "a" * 40}

    monkeypatch.setattr("modiff.custom_extension_source.resolve_hub_extension", resolve)

    async def scenario():
        lookup = asyncio.create_task(backend.custom_modules_resolve(request({"source": "example/block"})))
        try:
            async with asyncio.timeout(3):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
                lookup.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await lookup
                assert not backend._node_cache_lock.locked()
                assert not backend._node_cache_teardown_active
                assert store.list() == []
        finally:
            release.set()
        async with asyncio.timeout(3):
            while not finished.is_set():
                await asyncio.sleep(0.01)
        assert store.list() == []

    asyncio.run(scenario())


@pytest.mark.parametrize("body", [{}, {"source": "example/block", "consent": True}, {"source": 123}])
def test_source_resolution_api_rejects_invalid_requests(backend, body):
    import asyncio

    response = asyncio.run(backend.custom_modules_resolve(request(body)))
    assert response.status == 400
    assert json.loads(response.text)["error"] is True


@pytest.mark.parametrize("entry_name", ["block", "main"])
def test_two_modular_packages_keep_relative_helpers_isolated(store, tmp_path, monkeypatch, entry_name):
    import shutil
    from modules import MODULE_MAP

    fixture = Path(__file__).resolve().parents[1] / "examples/custom_nodes/ModularPrompt"
    try:
        for name, suffix in [("Example", " first"), ("Second", " second")]:
            source = tmp_path / name
            shutil.copytree(fixture, source)
            if entry_name != "block":
                (source / "block.py").rename(source / f"{entry_name}.py")
            config = source / "modular_config.json"
            config.write_text(config.read_text().replace("block.PromptSuffix", f"{entry_name}.PromptSuffix"))
            code = source / f"{entry_name}.py"
            code.write_text(
                code.read_text(encoding="utf-8").replace(
                    "        state.set(",
                    f"        from .{entry_name} import PromptSuffix\n        assert isinstance(self, PromptSuffix)\n        state.set(",
                ), encoding="utf-8"
            )
            code.write_text("from .helper import SUFFIX\n" + code.read_text(encoding="utf-8").replace('" — modular"', "SUFFIX"), encoding="utf-8")
            (source / "helper.py").write_text(f"SUFFIX = {suffix!r}\n")
            item = store.stage(kind="local", source=str(source), name=name)
            registry = store.enable(name, code_hash=item["codeHash"], consent=True)
            monkeypatch.setitem(MODULE_MAP, f"custom.{name}", registry)
        assert sys.modules["custom.Example.main"].Block("first").execute(text="test") == {"out_result": "test first"}
        assert sys.modules["custom.Second.main"].Block("second").execute(text="test") == {"out_result": "test second"}
        helper = store.path("Example") / "helper.py"
        helper.write_text("SUFFIX = ' fresh'\n")
        changed = store.inspect("Example")
        store.enable("Example", code_hash=changed["codeHash"], consent=True)
        assert sys.modules["custom.Example.main"].Block("first").execute(text="test") == {"out_result": "test fresh"}
        assert sys.modules["custom.Second.main"].Block("second").execute(text="test") == {"out_result": "test second"}
    finally:
        store.unload("Second")
        assert not any(getattr(finder, "prefix", "").startswith("custom.Second") for finder in sys.meta_path)


def test_lazy_import_uses_only_approved_bytes_even_if_source_changes(store, tmp_path, monkeypatch):
    from modules import MODULE_MAP

    source = source_node(tmp_path / "source")
    code = source / "main.py"
    code.write_text(
        code.read_text().replace(
            'return {"out": text.upper()}', 'from .helper import transform; return {"out": transform(text)}'
        )
    )
    (source / "helper.py").write_text("def transform(text): return text.upper()\n")
    item = store.stage(kind="local", source=str(source), name="Example")
    registry = store.enable("Example", code_hash=item["codeHash"], consent=True)
    monkeypatch.setitem(MODULE_MAP, "custom.Example", registry)
    (store.path("Example") / "helper.py").write_text('raise RuntimeError("unapproved edit")\n')
    # Direct Python call demonstrates the import snapshot. Normal graph dispatch
    # additionally rejects filesystem drift before executing even approved bytes.
    assert sys.modules["custom.Example.main"].Echo("lazy").execute("Hello") == {"out": "HELLO"}


def test_custom_browser_assets_require_the_same_approval_and_reject_traversal(store, tmp_path):
    source = source_node(tmp_path / "source")
    (source / "web").mkdir()
    (source / "web/field.js").write_text('export const label = "Field";\n')
    item = store.stage(kind="local", source=str(source), name="Example")
    with pytest.raises(ExtensionError, match="approval"):
        store.asset("Example", "field.js")
    store.enable("Example", code_hash=item["codeHash"], consent=True)
    assert store.asset("Example", "field.js").startswith(b"export")
    with pytest.raises(ExtensionError, match="path"):
        store.asset("Example", "../main.py")
    store.disable("Example")
    with pytest.raises(ExtensionError, match="approval"):
        store.asset("Example", "field.js")


def test_auto_accepts_approved_resource_roles_without_importing_or_granting_trust(store, tmp_path, monkeypatch):
    from modiff.workflow_auto_resource import build_workflow_auto_plan

    monkeypatch.setattr("modiff.custom_extensions.ExtensionStore", lambda: store)
    source = source_node(tmp_path / "source")
    (source / "modiff_extension.json").write_text('{"runtimeRole":"data"}')
    item = store.stage(kind="local", source=str(source), name="Example")
    graph = {
        "nodes": {"custom": {"module": "custom.Example", "action": "Echo", "params": {"text": {"value": "Hello"}}}},
        "paths": [["custom"]],
    }

    def plan():
        return build_workflow_auto_plan(
            graph,
            runtime_fingerprint={},
            local_models=[],
            data_dir=str(tmp_path),
            hardware={"accelerator": {}, "systemMemory": {}, "offloadDisk": {}},
        )

    assert not plan()["canAutoRun"]
    store.enable("Example", code_hash=item["codeHash"], consent=True)
    monkeypatch.setattr(store, "_load", lambda *_: pytest.fail("planning imported code"))
    assert plan()["canAutoRun"]
    assert plan()["schedule"] is None
    policy = store.path("Example") / "modiff_extension.json"
    policy.write_text('{"runtimeRole":"manual"}')
    assert not plan()["canAutoRun"]


def test_staging_never_copies_model_weights_or_source_approval(store, tmp_path):
    source = source_node(tmp_path / "source")
    weights = source / "model.safetensors"
    weights.write_bytes(b"preserved model")
    item = store.stage(kind="local", source=str(source), name="Example")
    assert "model.safetensors" not in {file["name"] for file in item["files"]}
    assert weights.read_bytes() == b"preserved model"
    assert not (Path(item["path"]) / "model.safetensors").exists()


@pytest.mark.parametrize("unsafe_path", [None, "C:/outside.py", "C:outside.py"])
def test_git_stages_exact_commit_without_checking_out_weights(store, tmp_path, monkeypatch, unsafe_path):
    import subprocess

    source = source_node(tmp_path / "git-source")
    (source / "model.safetensors").write_bytes(b"preserved fixture weights")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.DEVNULL).decode().strip()

    git("init")
    git("add", ".")
    git("-c", "user.name=Extension Test", "-c", "user.email=test@example.invalid", "commit", "-m", "Source fixture")
    revision = git("rev-parse", "HEAD")
    original = subprocess.run
    calls = []

    def transport(args, **kwargs):
        # Exercise real Git using a local fixture transport after production URL
        # validation, without relying on an external host or downloading models.
        args = [str(source) if arg == "https://example.invalid/fixture.git" else arg for arg in args]
        calls.append(args)
        result = original(args, **kwargs)
        if unsafe_path and "ls-tree" in args:
            result.stdout += b"100644 blob " + b"0" * 40 + b"\t" + unsafe_path.encode() + b"\0"
        return result

    monkeypatch.setattr("modiff.custom_extensions.subprocess.run", transport)
    if unsafe_path:
        with pytest.raises(ExtensionError, match="Invalid Git source path"):
            store.stage(kind="git", source="https://example.invalid/fixture.git", name="Example", revision=revision)
        assert not store.path("Example").exists()
        return
    item = store.stage(kind="git", source="https://example.invalid/fixture.git", name="Example", revision=revision)
    assert item["revision"] == revision and not item["enabled"]
    assert not any("checkout" in args for args in calls)
    assert not (store.path("Example") / "model.safetensors").exists()
    assert (source / "model.safetensors").read_bytes() == b"preserved fixture weights"
    # Git staging owns committed blob bytes, not the checkout's CRLF conversion.
    assert (store.path("Example") / "main.py").read_bytes() == subprocess.check_output(
        ["git", "-C", str(source), "show", f"{revision}:main.py"]
    )


def test_global_remote_code_disable_still_blocks_modular_activation(store, monkeypatch):
    fixture = Path(__file__).resolve().parents[1] / "examples/custom_nodes/ModularPrompt"
    item = store.stage(kind="local", source=str(fixture), name="Example")
    monkeypatch.setattr("diffusers.utils.dynamic_modules_utils.DIFFUSERS_DISABLE_REMOTE_CODE", True)
    with pytest.raises(ExtensionError, match="disabled globally"):
        store.enable("Example", code_hash=item["codeHash"], consent=True)
    assert not store.inspect("Example")["enabled"]


@pytest.mark.parametrize(
    "filename,content",
    [
        ("modular_config.json", "[]"),
        ("modular_config.json", '{"requirements":[{}]}'),
        ("modiff_extension.json", '{"runtimeRole":"fake"}'),
        ("modiff_extension.json", '{"runtimeRole":{}}'),
        ("pyproject.toml", '[project]\ndependencies="not an array"'),
    ],
)
def test_invalid_dependency_and_resource_metadata_never_imports(store, tmp_path, filename, content):
    source = source_node(tmp_path / "source")
    (source / filename).write_text(content)
    with pytest.raises(ExtensionError):
        store.stage(kind="local", source=str(source), name="Example")
    assert "custom.Example.main" not in sys.modules


def test_unchanged_approval_survives_startup_but_changed_source_does_not(store, tmp_path):
    item = store.stage(kind="local", source=str(source_node(tmp_path / "source")), name="Example")
    store.enable("Example", code_hash=item["codeHash"], consent=True)
    store.unload("Example")
    registry = {}
    store.load_enabled(registry)
    assert "custom.Example" in registry
    code = store.path("Example") / "main.py"
    code.write_text(code.read_text() + "\n# changed source\n")
    store.unload("Example")
    registry = {}
    store.load_enabled(registry)
    assert registry == {}
    assert "custom.Example.main" not in sys.modules


@pytest.mark.parametrize("declared_ports", [True, False, "collision"])
def test_modular_connected_weights_reuse_native_manager_and_survive_custom_release(
    backend, store, tmp_path, monkeypatch, declared_ports
):
    import asyncio
    import shutil
    from types import SimpleNamespace
    import torch
    from diffusers import ComponentsManager
    from modules import MODULE_MAP

    fixture = Path(__file__).resolve().parents[1] / "examples/custom_nodes/ModularPrompt"
    source = tmp_path / "component-block"
    shutil.copytree(fixture, source)
    code = source / "block.py"
    code.write_text(
        "import torch\nfrom diffusers.modular_pipelines import ComponentSpec\n"
        + code.read_text(encoding="utf-8")
        .replace(
            "    @property\n    def inputs(self):",
            '    @property\n    def expected_components(self):\n        return [ComponentSpec(name="weights", type_hint=torch.nn.Linear)]\n\n    @property\n    def inputs(self):',
        )
        .replace('state.get("text") + " — modular"', 'state.get("text") + str(int(pipeline.weights.weight[0, 0]))'),
        encoding="utf-8",
    )
    sidecar = source / "mellon_pipeline_config.json"
    data = json.loads(sidecar.read_text())
    contract = data["node_params"]["custom"]
    explicit = declared_ports is True
    if explicit:
        contract["model_input_names"] = ["weights"]
        contract["params"]["weights"] = {"type": "diffusers_auto_model", "display": "input"}
    else:
        del contract["model_input_names"]
        if declared_ports == "collision":
            contract["params"]["pipeline_components"] = {"type": "string", "default": "author value"}
    sidecar.write_text(json.dumps(data))
    item = store.stage(kind="local", source=str(source), name="Example")
    registry = store.enable("Example", code_hash=item["codeHash"], consent=True)
    port = "weights" if explicit else "modiff_pipeline_components" if declared_ports == "collision" else "pipeline_components"
    assert registry["Block"]["params"][port]["display"] == "input"
    if not explicit:
        assert registry["Block"]["params"][port]["type"] == "diffusers_modular_pipeline_components"
        assert port not in item["preview"]["nodes"]["Block"]["params"]
        assert store.inspect("Example")["codeHash"] == item["codeHash"]
    if declared_ports == "collision":
        assert registry["Block"]["params"]["pipeline_components"] == contract["params"]["pipeline_components"]
    monkeypatch.setitem(MODULE_MAP, "custom.Example", registry)
    backend.modules["custom.Example"] = registry
    manager = ComponentsManager()
    monkeypatch.setattr("modules.ModularDiffusers.components", manager)
    weights = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        weights.weight.fill_(3)
    comp_id = manager.add("weights", weights, collection="loader")
    value = {"model_id": comp_id} if explicit else {"weights": {"model_id": comp_id}}
    backend.node_cache["loader"] = SimpleNamespace(output={"weights": value}, _has_changed=False)
    node = {
        "module": "custom.Example",
        "action": "Block",
        "params": {"text": {"value": "value="}, port: {"sourceId": "loader", "sourceKey": "weights"}},
    }
    backend.execute_node("block", node, "test", quiet=True)
    assert backend.node_cache["block"].output == {"out_result": "value=3"}

    backend.execute_node("block", node, "test", quiet=True)
    assert not backend.node_cache["block"]._has_changed
    assert manager.components[comp_id] is weights
    asyncio.run(backend.delete_cache(request({"nodes": ["block"]})))
    assert manager.components[comp_id] is weights
    assert comp_id in manager.collections["loader"]
    backend.execute_node("block", node, "test", quiet=True)
    assert backend.node_cache["block"].output == {"out_result": "value=3"}

    # A real upstream call must reject missing/incompatible connected weights,
    # before the custom body tries to read them. Neither path loads defaults.
    from diffusers import ModularPipeline
    monkeypatch.setattr(ModularPipeline, "load_components", lambda *a, **k: pytest.fail("implicit model loading"))
    action = sys.modules["custom.Example.main"].Block
    with pytest.raises(ExtensionError, match="Connect loaded components.*weights"):
        action("missing").execute(text="value=")
    other = torch.nn.ReLU()
    other_id = manager.add("weights", other, collection="other-loader")
    invalid = {"model_id": other_id} if explicit else {"weights": {"model_id": other_id}}
    with pytest.raises(ValueError, match="expected torch.nn.modules.linear.Linear"):
        action("incompatible").execute(text="value=", **{port: invalid})
    assert manager.components[comp_id] is weights

    # Source reload retains the derived socket identity and reuses the loader's
    # real manager component, without rewriting its immutable sidecar.
    code = Path(item["path"]) / "block.py"
    code.write_text(code.read_text().replace('state.get("text") + str(', 'state.get("text") + "reloaded=" + str('))
    changed = store.inspect("Example")
    updated = store.enable("Example", code_hash=changed["codeHash"], consent=True)
    monkeypatch.setitem(MODULE_MAP, "custom.Example", updated)
    assert updated["Block"]["params"][port] == registry["Block"]["params"][port]
    assert sys.modules["custom.Example.main"].Block("reloaded").execute(text="value=", **{port: value}) == {
        "out_result": "value=reloaded=3"
    }


@pytest.mark.parametrize("model_type,denoiser", [
    ("FluxModularPipeline", "transformer"),
    ("StableDiffusionXLModularPipeline", "unet"),
])
def test_models_loader_publishes_managed_bundle_for_unscoped_pipeline(model_type, denoiser, monkeypatch):
    """Transport/ownership contract using tiny modules, not model qualification."""
    import torch
    from diffusers import ComponentsManager, ModularPipelineBlocks, ModelMixin
    from diffusers.modular_pipelines import ComponentSpec
    from modules.ModularDiffusers import loaders
    from modules.ModularDiffusers.route_state import require_component_binding

    class TinyModel(ModelMixin):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(1, 1)

    class TinyBlocks(ModularPipelineBlocks):
        @property
        def expected_components(self):
            return [ComponentSpec(name=name, type_hint=TinyModel) for name in
                    (denoiser, "vae", "scheduler", "controlnet")]

        def __call__(self, pipeline, state):
            raise AssertionError("Loader must not execute model blocks")

    manager = ComponentsManager()
    pipeline = TinyBlocks().init_pipeline(components_manager=manager, collection="bundle-loader")
    monkeypatch.setattr(loaders, "components", manager)
    monkeypatch.setattr(loaders.ModelsLoader, "_preflight_reviewed_builtin_selection",
                        lambda *a, **k: ("hub", "fixture/tiny", "a" * 40, "model_index.json", {}))
    monkeypatch.setattr(loaders, "_instantiate_reviewed_builtin_pipeline", lambda *a, **k: pipeline)
    loaded = []

    def load(pipeline, names, **kwargs):
        assert set(kwargs["required_names"]) == {denoiser, "vae", "scheduler"}
        for name in names:
            if name == "controlnet":
                continue  # An inactive optional component must stay unloaded.
            loaded.append(name)
            pipeline.update_components(**{name: TinyModel()})

    monkeypatch.setattr(loaders, "load_components_strict", load)
    node = loaders.ModelsLoader("bundle-loader")
    output = node.execute(model_type=model_type, repo_id={"source": "hub", "value": "fixture/tiny"},
                          device="cpu", dtype=torch.float32, auto_offload=False, offload_mode="none")
    bundle = output["pipeline_components"]
    assert set(loaded) == {denoiser, "vae", "scheduler"}
    assert "controlnet" not in bundle
    for name in loaded:
        assert manager.get_one(component_id=bundle[name]["model_id"]) is getattr(pipeline, name)
        assert bundle[name]["model_id"] in manager.collections["bundle-loader"]
    assert bundle["vae"]["model_id"] == output["vae_out"]["model_id"]
    require_component_binding(bundle, label="custom models", expected_model_type=model_type,
                              expected_role="pipeline_components")


@pytest.mark.parametrize("dtype,force_upcast", [("float32", True), ("float16", True), ("float16", False)])
def test_modular_image_reconstruction_uses_connected_vae_and_native_dispatch(
    backend, store, monkeypatch, dtype, force_upcast
):
    import numpy as np
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from diffusers import AutoencoderKL, ComponentsManager
    from modules import MODULE_MAP

    source = Path(__file__).resolve().parents[1] / "examples/custom_nodes/ModularImageReconstruction"
    item = store.stage(kind="local", source=str(source), name="Example")
    assert item["runtimeRole"] == "connected_components"
    registry = store.enable("Example", code_hash=item["codeHash"], consent=True)
    monkeypatch.setitem(MODULE_MAP, "custom.Example", registry)
    backend.modules["custom.Example"] = registry
    manager = ComponentsManager()
    monkeypatch.setattr("modules.ModularDiffusers.components", manager)
    original_dtype = getattr(torch, dtype)
    vae = AutoencoderKL(block_out_channels=(8,), norm_num_groups=4, latent_channels=4,
                        force_upcast=force_upcast).eval().to(dtype=original_dtype)
    comp_id = manager.add("vae", vae, collection="loader")
    original = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
    backend.node_cache["loader"] = SimpleNamespace(
        output={"models": {"vae": {"model_id": comp_id}}}, _has_changed=False,
    )
    node = {"module": "custom.Example", "action": "Block", "params": {
        "pipeline_components": {"sourceId": "loader", "sourceKey": "models"},
        "image": {"value": original}, "amount": {"value": "1.0"},
    }}
    calls = []
    handle = vae.register_forward_hook(lambda *args: calls.append(True))
    observed = []
    def check_precision(module, args):
        expected = torch.float32 if force_upcast and original_dtype == torch.float16 else original_dtype
        assert module.dtype == expected and args[0].dtype == expected
        observed.append(expected)
    pre_handle = vae.register_forward_pre_hook(check_precision)
    try:
        backend.execute_node("reconstruction", node, "test", quiet=True)
        first = backend.node_cache["reconstruction"].output["out_images"][0]
        assert first.size == original.size and first.mode == "RGB"
        assert first.tobytes() != original.tobytes()
        backend.execute_node("reconstruction", node, "test", quiet=True)
        assert len(calls) == 1  # Normal NodeBase cache, not a custom executor.
        node["params"]["amount"]["value"] = "0.0"
        backend.execute_node("reconstruction", node, "test", quiet=True)
        restored = backend.node_cache["reconstruction"].output["out_images"][0]
        assert restored.tobytes() == original.tobytes()
        assert len(calls) == 2
        assert len(observed) == 2 and vae.dtype == original_dtype
        assert manager.components[comp_id] is vae
        assert comp_id in manager.collections["loader"]
        assert all(not parameter.requires_grad or parameter.grad is None for parameter in vae.parameters())
    finally:
        handle.remove()
        pre_handle.remove()

    def fail_forward(*args):
        raise RuntimeError("test forward failure")
    failing_handle = vae.register_forward_pre_hook(fail_forward)
    try:
        node["params"]["amount"]["value"] = "0.25"
        with pytest.raises(RuntimeError, match="test forward failure"):
            backend.execute_node("reconstruction", node, "test", quiet=True)
        assert vae.dtype == original_dtype
        assert manager.components[comp_id] is vae
    finally:
        failing_handle.remove()


def test_invalid_approval_records_report_disabled_diagnostics(store, tmp_path):
    store.stage(kind="local", source=str(source_node(tmp_path / "source")), name="Example")
    (store.root / ".extensions.json").write_text('{"Example": []}')
    item = store.list()[0]
    assert item["status"] == "error"
    assert not item["enabled"]
    assert "Invalid extension approval file" in item["diagnostic"]
    registry = {}
    store.load_enabled(registry)
    assert not registry
    assert "custom.Example.main" not in sys.modules


def test_custom_modular_loader_requires_pins_and_reuses_native_components(backend, store, tmp_path, monkeypatch):
    from diffusers import AutoencoderKL, ComponentsManager
    from modules import MODULE_MAP
    from PIL import Image
    import torch

    source = Path(__file__).resolve().parents[1] / 'examples/custom_nodes/ModularImageReconstruction'
    item = store.stage(kind='local', source=str(source), name='Example')
    assert 'LoadModels' not in item['preview']['nodes']  # No Python imports during inspection.
    registry = store.enable('Example', code_hash=item['codeHash'], consent=True)
    assert registry['LoadModels']['params']['pipeline_components']['type'] == 'diffusers_modular_pipeline_components'
    assert registry['LoadModels']['params']['source__vae__repo']['fieldOptions']['filter'] == {
        'hub': {'className': ['AutoencoderKL']},
    }
    monkeypatch.setitem(MODULE_MAP, 'custom.Example', registry)
    backend.modules['custom.Example'] = registry
    manager = ComponentsManager()
    monkeypatch.setattr('modules.ModularDiffusers.components', manager)
    weights = tmp_path / 'models--fixture--vae' / 'snapshots' / ('a' * 40)
    AutoencoderKL(block_out_channels=(8,), norm_num_groups=4, latent_channels=4).save_pretrained(weights)
    lookups = []
    def cached_snapshot(repo_id, *, revision, local_files_only, allow_patterns):
        assert repo_id == 'fixture/vae' and revision == 'a' * 40 and local_files_only is True
        assert allow_patterns == ['*config.json']
        lookups.append(repo_id)
        return str(weights)
    monkeypatch.setattr('huggingface_hub.snapshot_download', cached_snapshot)
    original_load = AutoencoderKL.from_pretrained
    loads = []
    def load_cached(repo, **kwargs):
        assert repo == 'fixture/vae' and kwargs['revision'] == 'a' * 40
        assert kwargs['subfolder'] == ''
        assert kwargs['local_files_only'] and kwargs['trust_remote_code'] is False
        assert kwargs['use_safetensors'] and kwargs['weights_only']
        loads.append(repo)
        return original_load(weights, **kwargs)
    monkeypatch.setattr(AutoencoderKL, 'from_pretrained', load_cached)
    node = {'module': 'custom.Example', 'action': 'LoadModels', 'params': {
        'source__vae__repo': {'value': {'source': 'hub', 'value': 'fixture/vae'}},
        'source__vae__revision': {'value': 'main'},
        'device': {'value': 'cpu:0'}, 'dtype': {'value': 'float32'}, 'offload_mode': {'value': 'none'},
    }}
    with pytest.raises(ValueError, match='40-character'):
        backend.execute_node('custom-loader', node, 'test', quiet=True)
    assert not lookups and not manager.components
    node['params']['source__vae__revision']['value'] = 'a' * 40
    backend.execute_node('custom-loader', node, 'test', quiet=True)
    bundle = backend.node_cache['custom-loader'].output['pipeline_components']
    vae_id = bundle['vae']['model_id']
    vae = manager.get_one(component_id=vae_id)
    assert isinstance(vae, AutoencoderKL) and vae.dtype == torch.float32
    backend.execute_node('custom-loader', node, 'test', quiet=True)
    assert not backend.node_cache['custom-loader']._has_changed
    backend.execute_node('second-loader', node, 'test', quiet=True)
    assert backend.node_cache['second-loader'].output['pipeline_components']['vae']['model_id'] == vae_id
    assert vae_id in manager.collections['custom-loader'] and vae_id in manager.collections['second-loader']
    assert len(loads) == 1
    consumer = {'module': 'custom.Example', 'action': 'Block', 'params': {
        'pipeline_components': {'sourceId': 'custom-loader', 'sourceKey': 'pipeline_components'},
        'image': {'value': Image.new('RGB', (16, 16), 'red')}, 'amount': {'value': 0.5},
    }}
    backend.execute_node('custom-consumer', consumer, 'test', quiet=True)
    assert backend.node_cache['custom-consumer'].output['out_images'][0].size == (16, 16)
    # Metadata drift must not hit either the node cache or another owner's old
    # component, while the old owner keeps its exact loaded object.
    config = json.loads((weights / 'config.json').read_text())
    config['scaling_factor'] = 0.25
    (weights / 'config.json').write_text(json.dumps(config))
    backend.execute_node('custom-loader', node, 'test', quiet=True)
    replacement_id = backend.node_cache['custom-loader'].output['pipeline_components']['vae']['model_id']
    assert replacement_id != vae_id and len(loads) == 2
    assert manager.get_one(component_id=vae_id) is vae
    assert manager.get_one(component_id=replacement_id).config.scaling_factor == 0.25
    # Validation must precede cached reuse, including selectors a graph cannot authorize.
    node['params']['source__vae__repo']['value'] = {'source': 'local', 'value': str(weights)}
    with pytest.raises(ValueError, match='Hub'):
        backend.execute_node('custom-loader', node, 'test', quiet=True)
    assert manager.get_one(component_id=vae_id) is vae
    backend._release_node_modular_components(['custom-loader'])
    assert replacement_id not in manager.components
    assert manager.get_one(component_id=vae_id) is vae


def test_custom_model_supplier_requires_manual_policy_even_for_connected_source(store, tmp_path, monkeypatch):
    from modiff.workflow_auto_resource import build_workflow_auto_plan
    source = Path(__file__).resolve().parents[1] / 'examples/custom_nodes/ModularImageReconstruction'
    item = store.stage(kind='local', source=str(source), name='Example')
    store.enable('Example', code_hash=item['codeHash'], consent=True)
    assert store.inspect('Example')['nodeCount'] == 2
    assert store.inspect('Example')['nodes'] == ['Block', 'LoadModels']
    monkeypatch.setattr('modiff.custom_extensions.ExtensionStore', lambda: store)
    monkeypatch.setattr(store, '_load', lambda *_: pytest.fail('planning must not import custom Python'))
    graph = {'nodes': {'models': {'module': 'custom.Example', 'action': 'LoadModels', 'params': {}}}, 'paths': [['models']]}
    result = build_workflow_auto_plan(graph, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path),
                                     hardware={'accelerator': {}, 'systemMemory': {}, 'offloadDisk': {}})
    assert not result['canAutoRun']
    assert 'Custom memory policy' in ' '.join(result['issues'])


@pytest.mark.parametrize('field,value', [
    ('revision', ''), ('revision', 'main'), ('revision', 'A' * 40),
    ('repo', {'source': 'hub', 'value': '../outside'}),
    ('subfolder', '../outside'), ('variant', '../weights'),
])
def test_custom_model_source_validation_precedes_cache_lookup(store, monkeypatch, field, value):
    from modules import MODULE_MAP
    source = Path(__file__).resolve().parents[1] / 'examples/custom_nodes/ModularImageReconstruction'
    item = store.stage(kind='local', source=str(source), name='Example')
    registry = store.enable('Example', code_hash=item['codeHash'], consent=True)
    monkeypatch.setitem(MODULE_MAP, 'custom.Example', registry)
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda *a, **k: pytest.fail('invalid selector reached cache'))
    args = {'source__vae__repo': {'source': 'hub', 'value': 'fixture/vae'},
            'source__vae__revision': 'a' * 40, 'source__vae__subfolder': '', 'source__vae__variant': ''}
    args['source__vae__' + field] = value
    with pytest.raises(ValueError):
        sys.modules['custom.Example.main'].LoadModels('invalid-selection')(**args)
