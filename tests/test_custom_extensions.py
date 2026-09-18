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


def test_modular_connected_weights_reuse_native_manager_and_survive_custom_release(
    backend, store, tmp_path, monkeypatch
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
    contract["model_input_names"] = ["weights"]
    contract["params"]["weights"] = {"type": "diffusers_auto_model", "display": "input"}
    sidecar.write_text(json.dumps(data))
    item = store.stage(kind="local", source=str(source), name="Example")
    registry = store.enable("Example", code_hash=item["codeHash"], consent=True)
    monkeypatch.setitem(MODULE_MAP, "custom.Example", registry)
    backend.modules["custom.Example"] = registry
    manager = ComponentsManager()
    monkeypatch.setattr("modules.ModularDiffusers.components", manager)
    weights = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        weights.weight.fill_(3)
    comp_id = manager.add("weights", weights, collection="loader")
    backend.node_cache["loader"] = SimpleNamespace(output={"weights": {"model_id": comp_id}}, _has_changed=False)
    node = {
        "module": "custom.Example",
        "action": "Block",
        "params": {"text": {"value": "value="}, "weights": {"sourceId": "loader", "sourceKey": "weights"}},
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
