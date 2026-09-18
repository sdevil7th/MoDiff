from copy import deepcopy
import json

import pytest

from modiff import service_package as service

REGISTRY = {
    "modules.Primitive": {
        "TextValue": {
            "params": {
                "text": {"type": "string", "display": "text"},
                "output": {"display": "output", "type": "string"},
            }
        },
        "DataViewer": {
            "params": {
                "value": {"display": "input", "type": "any"},
                "preview": {"display": "ui_text", "dataSource": "output"},
            }
        },
    }
}
GRAPH = {
    "sid": "browser",
    "nodes": {
        "prompt": {
            "module": "modules.Primitive",
            "action": "TextValue",
            "params": {"text": {"value": "private prompt"}},
        },
        "preview": {
            "module": "modules.Primitive",
            "action": "DataViewer",
            "params": {
                "value": {"sourceId": "prompt", "sourceKey": "output"},
                "preview": {"display": "ui_text", "sourceKey": "output"},
            },
        },
    },
    "paths": [["prompt", "preview"]],
}
INTERFACE = {
    "inputs": {"prompt": [{"nodeId": "prompt", "field": "text"}]},
    "outputs": {"text": [{"nodeId": "preview", "field": "preview"}]},
}
CONTRACT = {
    "backend": {"fingerprint": "test"},
    "packages": {"diffusers": "0.test"},
    "customNodes": [],
    "optionalProfiles": [],
}


def build(graph=None, interface=None):
    return service.build_package(
        deepcopy(graph or GRAPH), deepcopy(interface or INTERFACE), registry=REGISTRY, contract=CONTRACT
    )


def prepare(package, values=None, contract=None):
    return service.prepare_package(
        package,
        values or {"prompt": "public input"},
        registry=REGISTRY,
        contract=contract or CONTRACT,
        sid="service_test",
    )


def test_roundtrip_preserves_graph_and_omits_session_snapshots_and_input_defaults():
    graph = deepcopy(GRAPH)
    graph["runtimeHints"] = {
        "workflowSnapshot": {"path": "/home/private"},
        "workflowTabId": "private",
        "device": "cpu",
    }
    package = build(graph)
    assert "private" not in json.dumps(package)
    assert "sid" not in package["graph"]
    result = prepare(package)
    expected = deepcopy(GRAPH)
    expected["sid"] = "service_test"
    expected["nodes"]["prompt"]["params"]["text"]["value"] = "public input"
    assert service.graph_material(result) == service.graph_material(expected)
    assert result["runtimeHints"] == {"device": "cpu"}
    assert package["graph"]["nodes"]["prompt"]["params"]["text"]["value"] is None
    assert result["servicePackage"]["package"] == package


@pytest.mark.parametrize(
    "value",
    [
        "/home/alice/models",
        "C:\\Users\\Alice\\model",
        "hf_" + "a" * 30,
        "https://name:password@example.org/a",
        "@data/private.png",
    ],
)
def test_local_paths_and_known_credentials_never_export(value):
    graph = deepcopy(GRAPH)
    graph["nodes"]["prompt"]["params"]["text"]["value"] = value
    with pytest.raises(ValueError, match="credential/local path"):
        build(graph, {**INTERFACE, "inputs": {}})
    assert value not in json.dumps(build(graph))


def test_bound_inputs_cannot_change_model_identity_or_connected_fields():
    interface = deepcopy(INTERFACE)
    interface["inputs"]["prompt"] = [{"nodeId": "preview", "field": "value"}]
    with pytest.raises(ValueError, match="binding"):
        build(interface=interface)
    interface = deepcopy(INTERFACE)
    interface["inputs"]["duplicate"] = interface["inputs"]["prompt"]
    with pytest.raises(ValueError, match="duplicated"):
        build(interface=interface)


def test_tampering_and_runtime_drift_require_new_export():
    package = build()
    package["graph"]["nodes"]["prompt"]["params"]["text"]["value"] = "tampered"
    with pytest.raises(ValueError, match="content hash"):
        prepare(package)
    with pytest.raises(ValueError, match="requirements changed"):
        prepare(build(), contract={**CONTRACT, "packages": {"diffusers": "different"}})
    with pytest.raises(ValueError, match="requires string"):
        prepare(build(), {"prompt": 12})
    with pytest.raises(ValueError, match="exactly"):
        prepare(build(), {"prompt": "x", "extra": True})


def test_auto_receipt_is_rebound_to_invocation_values():
    graph = deepcopy(GRAPH)
    graph["runtimeHints"] = {"resourceMode": "auto", "workflowAutoPlan": {"graphHash": "stale"}}
    package = build(graph)
    result = prepare(package)
    assert result["runtimeHints"]["workflowAutoPlan"]["graphHash"] == service.workflow_graph_hash(result)
    assert "workflowAutoPlan" not in package["graph"]["runtimeHints"]


def test_outputs_are_exact_task_scoped_and_do_not_copy_private_snapshots():
    package = build()
    receipt = {
        "outputs": [
            {"taskId": "old", "nodeId": "preview", "fieldKey": "preview", "value": "old"},
            {
                "taskId": "new",
                "nodeId": "preview",
                "fieldKey": "preview",
                "value": ["text"],
                "graphSnapshot": {"secret": True},
            },
        ]
    }
    assert service.service_outputs(package, receipt, "new") == {
        "taskId": "new",
        "outputs": {"text": [{"value": ["text"]}]},
    }
    with pytest.raises(ValueError, match="did not persist"):
        service.service_outputs(package, receipt, "missing")


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b"[]" * 10, b"{" * 10000])
def test_json_boundary_rejects_duplicate_nonfinite_and_malformed_data(raw):
    with pytest.raises(ValueError):
        service.load_json(raw)


def test_unknown_model_requires_immutable_revision_and_never_downloads(monkeypatch):
    graph = deepcopy(GRAPH)
    node = graph["nodes"]["prompt"]
    node["params"]["repo_id"] = {"value": {"source": "hub", "value": "example/model"}}
    from modiff import model_artifact_catalog

    monkeypatch.setattr(model_artifact_catalog, "catalog_repository_pin", lambda _: None)
    with pytest.raises(ValueError, match="immutable"):
        build(graph)
    node["params"]["revision"] = {"value": "a" * 40}
    package = build(graph)
    assert package["requirements"]["models"] == [
        {"location": "prompt.repo_id", "repository": "example/model", "revision": "a" * 40}
    ]


def test_execution_guard_rechecks_contract_and_rejects_modified_prepared_graph():
    from modiff.service_api import ServiceAPI

    class Server(ServiceAPI):
        modules = REGISTRY
        contract = CONTRACT

        def _service_contract(self, graph):
            return self.contract

        def _coerce_runtime_hints(self, hints):
            return hints

    server = Server()
    graph = prepare(build())
    server._validate_service_graph(graph)
    with pytest.raises(ValueError, match="envelope"):
        server._validate_service_graph({**graph, "servicePackage": None})
    graph["nodes"]["prompt"]["params"]["text"]["value"] = "changed after preparation"
    with pytest.raises(ValueError, match="modified"):
        server._validate_service_graph(graph)
    server.contract = {**CONTRACT, "customNodes": [{"codeHash": "changed"}]}
    with pytest.raises(ValueError, match="requirements changed"):
        server._validate_service_graph(prepare(build()))


def test_service_api_rejects_malformed_and_duplicate_keys_without_execution():
    import asyncio
    from types import SimpleNamespace
    from modiff.service_api import ServiceAPI

    class Content:
        def __init__(self, raw):
            self.raw = raw

        async def iter_chunked(self, _size):
            yield self.raw

    server = ServiceAPI()
    server.modules = REGISTRY
    server._service_contract = lambda graph: CONTRACT

    async def post(body):
        response = await server.service_package(SimpleNamespace(content=Content(body)))
        return response.status, json.loads(response.text)

    async def scenario():
        status, body = await post(json.dumps({"operation": "build", "graph": GRAPH, "interface": INTERFACE}).encode())
        assert status == 200 and body["package"] == build()
        for raw in (b'{"operation":"inspect","operation":"build"}', b'{"operation":"prepare","package":[]}', b"[]"):
            status, body = await post(raw)
            assert status == 400 and body["error"] is True

    asyncio.run(scenario())


def test_cli_refuses_remote_redirect_and_credential_origins_without_connecting(monkeypatch):
    from modiff.service import request

    monkeypatch.setattr("http.client.HTTPConnection", lambda *a, **k: pytest.fail("unexpected network"))
    for origin in (
        "https://127.0.0.1",
        "http://example.com",
        "http://user:secret@127.0.0.1",
        "http://127.0.0.1/private",
        "http://localhost",
    ):
        with pytest.raises(ValueError, match="loopback"):
            request(origin, "/queue")


def test_cli_timeout_reports_exact_task_without_resubmitting(monkeypatch, capsys):
    from modiff import service as client

    calls = []

    def request(server, path, body=None):
        calls.append(path)
        return (
            {"graph": GRAPH}
            if path == "/service_package"
            else {"task_id": "owned_task"}
            if path == "/graph"
            else {"recent": []}
        )

    times = iter([0, 0, 2])
    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="owned_task.*remains queued/running"):
        client.run("http://127.0.0.1:8088", build(), {"prompt": "x"}, timeout=1)
    assert calls.count("/graph") == 1
    assert "owned_task" in capsys.readouterr().err


def test_preview_must_be_present_in_export_and_not_hidden():
    graph = deepcopy(GRAPH)
    graph["nodes"]["preview"]["params"].pop("preview")
    assert service.inspect_graph(graph, REGISTRY)["outputs"] == []
    with pytest.raises(ValueError, match="binding"):
        build(graph)


def test_media_projection_keeps_all_durable_items_without_private_paths():
    receipt = {
        "outputs": [
            {
                "taskId": "run",
                "nodeId": "preview",
                "fieldKey": "preview",
                "mediaItems": [
                    {"taskId": "run", "index": 0, "url": "/file?file=a", "backendPath": "/home/private/a"},
                    {"taskId": "run", "index": 1, "url": "/file?file=b", "graphSnapshot": {"private": True}},
                    {"taskId": "other", "url": "/wrong-run"},
                ],
            }
        ]
    }
    result = service.service_outputs(build(), receipt, "run")
    assert result["outputs"]["text"][0]["mediaItems"] == [
        {"index": 0, "url": "/file?file=a"},
        {"index": 1, "url": "/file?file=b"},
    ]
    assert "private" not in json.dumps(result)


def test_auxiliary_repositories_never_inherit_the_base_models_revision(monkeypatch):
    graph = deepcopy(GRAPH)
    graph["nodes"]["prompt"]["params"].update(
        {
            "model_id": {"value": "example/base"},
            "revision": {"value": "a" * 40},
            "lora": {"value": [{"repo_id": "example/auxiliary"}]},
        }
    )
    monkeypatch.setattr("modiff.model_artifact_catalog.catalog_repository_pin", lambda _: None)
    with pytest.raises(ValueError, match="immutable"):
        build(graph)
    graph["nodes"]["prompt"]["params"]["lora"]["value"][0]["revision"] = "b" * 40
    assert [pin["revision"] for pin in build(graph)["requirements"]["models"]] == ["a" * 40, "b" * 40]


def test_runtime_manifest_observes_active_overlay_first_and_omits_extension_paths(monkeypatch):
    from types import SimpleNamespace
    from modiff import runtime_profile

    monkeypatch.setattr(runtime_profile, "read_state", lambda _: {"profile": "cpu", "lock_digest": "reviewed"})
    monkeypatch.setattr(
        runtime_profile,
        "load_manifest",
        lambda: {"profiles": {"cpu": {"requirements": "requirements/profiles/cpu.txt"}}},
    )
    monkeypatch.setattr(runtime_profile, "lock_digest", lambda *args, **kw: "reviewed")
    monkeypatch.setattr(
        "modiff.optional_runtime_execution.graph_optional_runtime_requirement",
        lambda _: {"profileIds": ["example-overlay"]},
    )
    monkeypatch.setattr(
        service.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": "Example_Package"}, version="2.0"),
            SimpleNamespace(metadata={"Name": "example-package"}, version="1.0"),
        ],
    )
    graph = deepcopy(GRAPH)
    graph["nodes"]["prompt"]["module"] = "custom.Example"
    store = SimpleNamespace(
        require_enabled=lambda _: {
            "moduleKey": "custom.Example",
            "codeHash": "reviewed",
            "revision": None,
            "dependencies": [],
            "path": "/home/private/source",
        }
    )
    manifest = service.runtime_contract(graph, {"gitCommit": "a" * 40, "fingerprint": "source"}, store)
    assert manifest["packages"] == {"example-package": "2.0"}
    assert manifest["optionalProfiles"] == ["example-overlay"]
    assert manifest["customNodes"][0]["codeHash"] == "reviewed"
    assert "private" not in json.dumps(manifest)
