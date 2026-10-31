"""No-weight receipt protocol checks using the unchanged normal DDIM export.

The fixture is the normal frontend export of the checked-in canonical creator,
not a substitute workflow. Library allocation/inference are mocked; these tests
are not runtime, hardware, output quality or release qualification.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from modiff.execution_input_provenance import bind_generation_input_origins, build_resolved_execution_inputs


def ddim_graph():
    fixture = json.loads((Path(__file__).parent / "fixtures/ddim_unconditional_api_graph.json").read_text())
    source = Path(__file__).parents[1] / "data/graphs/studio/ddim-pipeline/unconditional-image.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == fixture["canonicalSourceSha256"]
    return fixture["apiGraph"]


def run_ddim(monkeypatch, graph=None):
    from modules.DiffusersImage import main as image
    from modules.DiffusersRuntime import main as runtime
    from modules.Image.main import Preview
    from modiff.modelstore import modelstore

    graph = graph or ddim_graph()
    calls = {}

    class DDIMPipeline:
        _execution_device = "cpu"

        def __call__(self, *, batch_size, num_inference_steps, generator, output_type, return_dict, eta):
            calls["generate"] = dict(
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                seed=generator.initial_seed(),
                output_type=output_type,
                eta=eta,
            )
            return SimpleNamespace(images=[Image.new("RGB", (32, 32))])

    def load(_cls, repo, **kwargs):
        calls["load"] = dict(repo=repo, **kwargs)
        return DDIMPipeline()

    def configure(_pipeline, recipe):
        calls["runtime"] = recipe
        return {
            "attention": {"requested": recipe["attention_backend"], "applied": ["unet"]},
            "vae": {"applied": [], "unsupported": ["slicing", "tiling"]},
        }

    monkeypatch.setattr(image, "pipeline_class_from_name", lambda _name: DDIMPipeline)
    monkeypatch.setattr(image, "load_cached_image_component", load)
    monkeypatch.setattr(image, "local_files_only", lambda _repo: True)
    monkeypatch.setattr(image, "apply_pipeline_offload", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "apply_execution_recipe_to_pipeline", configure)
    monkeypatch.setattr(runtime, "enable_parallel_weight_loading", lambda: False)
    monkeypatch.setattr(modelstore, "is_hf_cached", lambda _repo: True)
    types = {
        "LoadPipeline": image.LoadPipeline,
        "UnconditionalGenerate": image.UnconditionalGenerate,
        "PipelineQuantizationConfigV2": runtime.PipelineQuantizationConfigV2,
        "DiffusersExecutionRecipe": runtime.DiffusersExecutionRecipe,
        "Preview": Preview,
    }
    records, outputs, instances = {}, {}, {}
    for node_id in graph["paths"][0]:
        definition = graph["nodes"][node_id]
        instance = types[definition["action"]](node_id)
        # This CPU test environment has no CUDA option. Model the published
        # NVIDIA field schema without changing the actual canonical request.
        instance.default_params = deepcopy(instance.default_params)
        if "options" in instance.default_params.get("device", {}):
            instance.default_params["device"]["options"] = ["cpu:0", "cuda:0"]
        instance.progress = lambda *a, **kw: None
        instance.mm_add = lambda *a, **kw: None
        values = {
            key: (outputs[param["sourceId"]][param["sourceKey"]] if param.get("sourceId") else param["value"])
            for key, param in definition["params"].items()
            if param.get("sourceId") or "value" in param
        }
        outputs[node_id] = instance(**values)
        instances[node_id] = instance
        records[node_id] = bind_generation_input_origins(node_id, definition, instance._execution_input_record)
    receipt = build_resolved_execution_inputs(
        graph, records, task_id="protocol-only-ddim", attempt_index=0, node_id="node-05"
    )
    return receipt, calls, instances, outputs


def test_real_ddim_export_has_complete_native_literal_accounting(monkeypatch):
    graph = ddim_graph()
    receipt, calls, _, _ = run_ddim(monkeypatch, graph)
    assert len(receipt["nodes"]) == 5
    rows = {row["nodeId"]: row for row in receipt["nodes"]}
    for node_id, row in rows.items():
        accounting = row["fieldAccounting"]
        assert accounting["schemaVersion"] == 1
        assert accounting["complete"], accounting
        expected = {
            key
            for key, param in graph["nodes"][node_id]["params"].items()
            if "value" in param and not param.get("sourceId")
        }
        assert expected <= accounting["fields"].keys()
    generated = rows["node-04"]["fieldAccounting"]["fields"]
    for key in ("batch_size", "num_inference_steps", "seed", "eta", "output_type"):
        assert generated[key]["disposition"] == "consumed"
        assert generated[key]["value"] == calls["generate"][key]
    assert generated["class_label"]["disposition"] == "inactive"
    assert generated["image_contract"]["disposition"] == "metadata"
    assert rows["node-01"]["fieldAccounting"]["fields"]["device_map"]["value"] == calls["load"]["device_map"] == "cuda"
    assert rows["node-01"]["fieldAccounting"]["fields"]["enable_vae_slicing"]["disposition"] == "superseded"
    assert rows["node-01"]["fieldAccounting"]["fields"]["enable_vae_slicing"]["value"] is False
    assert rows["node-05"]["fieldAccounting"]["fields"]["export"]["value"] == "0"
    assert rows["node-05"]["fieldAccounting"]["fields"]["device"]["disposition"] == "inactive"
    producer = rows["node-03"]["fieldAccounting"]["outputProjections"]["execution_recipe"]
    consumer = rows["node-01"]["fieldAccounting"]["inputProjections"]["execution_recipe"]
    assert consumer["sha256"] == producer["sha256"]
    assert consumer["sourceNodeId"] == "node-03" and consumer["sourcePortId"] == "execution_recipe"
    assert (
        rows["node-02"]["fieldAccounting"]["outputProjections"]["quantization_config"]["sha256"]
        == (rows["node-03"]["fieldAccounting"]["inputProjections"]["quantization_config"]["sha256"])
    )


def test_unknown_literal_and_stale_contract_hold_evidence_without_changing_execution(monkeypatch):
    graph = ddim_graph()
    graph["nodes"]["node-04"]["params"]["unreviewed_control"] = {"value": "private-unknown-value"}
    graph["nodes"]["node-04"]["params"]["unreviewed_connection"] = {
        "sourceId": "node-03",
        "sourceKey": "execution_recipe",
    }
    graph["nodes"]["node-04"]["params"]["image_contract"]["value"]["pipelineClass"] = "ChangedPipeline"
    receipt, calls, _, outputs = run_ddim(monkeypatch, graph)
    assert outputs["node-05"]["filtered"].size == (32, 32)
    assert calls["generate"]["eta"] == 0
    accounting = next(row["fieldAccounting"] for row in receipt["nodes"] if row["nodeId"] == "node-04")
    assert not accounting["complete"]
    assert accounting["omittedFields"]["image_contract"] == "metadata-contract-mismatch"
    assert "unreviewed_control" in accounting["unaccountedFields"]
    assert "unreviewed_connection" in accounting["unaccountedFields"]
    assert "private-unknown-value" not in json.dumps(receipt)


def test_recipe_projection_is_bound_to_actual_configuration_and_no_opaque_values(monkeypatch):
    from modiff.execution_input_provenance import execution_recipe_projection, disabled_quantization_projection

    _, _, _, outputs = run_ddim(monkeypatch)
    recipe = outputs["node-03"]["execution_recipe"]
    original = execution_recipe_projection(recipe)
    assert execution_recipe_projection({**recipe, "device": "cpu:0"})["sha256"] != original["sha256"]
    for changed in (
        {**recipe, "max_memory": {"cpu": "1GiB"}},
        {**recipe, "cache_options": {"unreviewed": "private-value"}},
        {**recipe, "quantization_config": object()},
        {**recipe, "unexpected": "private-value"},
    ):
        assert execution_recipe_projection(changed) is None
    assert (
        disabled_quantization_projection(
            {"schema_version": 1, "backend": "none", "disabled": True, "token": "private-value"}
        )
        is None
    )


def test_accounting_bounds_and_scope_never_serialize_unknown_objects():
    from modiff.execution_input_provenance import capture_field_accounting, disabled_quantization_projection

    class Opaque:
        def __str__(self):
            raise AssertionError("opaque runtime object must not be stringified")

    scope = {"module": "modules.Image", "action": "Preview"}
    accounting = capture_field_accounting(
        scope, {"export": Opaque(), "token": "private-value"}, consumed={"export": Opaque()}
    )
    assert accounting["omittedFields"] == {"export": "unsupported-value"}
    assert not accounting["complete"] and "private-value" not in json.dumps(accounting)
    assert capture_field_accounting({"module": "Unknown", "action": "Preview"}, {}, consumed={}) is None
    for value in ("x" * 16385, [0] * 17, {"secret": "private-value"}, float("nan")):
        assert not capture_field_accounting(scope, {"export": value}, consumed={"export": value})["complete"]
    malicious = disabled_quantization_projection(None)
    malicious["values"]["token"] = "private-value"
    result = capture_field_accounting(
        {"module": "DiffusersRuntime", "action": "PipelineQuantizationConfigV2"},
        {"backend": "none"},
        consumed={"backend": "none"},
        output_projections={"quantization_config": malicious},
    )
    assert not result["complete"] and "private-value" not in json.dumps(result)
    missing = capture_field_accounting(
        {"module": "DiffusersRuntime", "action": "PipelineQuantizationConfigV2"},
        {"backend": "none"},
        consumed={"backend": "none"},
    )
    assert missing["omittedFields"] == {"quantization_config": "missing-projection"}
    assert not missing["complete"]


def test_accounting_size_limits_do_not_drop_or_modify_v1_fields():
    from modiff.execution_input_provenance import capture_field_accounting, capture_generation_inputs

    node = {"module": "modules.DiffusersRuntime", "action": "PipelineQuantizationConfigV2", "params": {}}
    values = {"excluded_modules": "x" * 16384, "component_overrides": "y" * 16384}
    accounting = capture_field_accounting(node, values, consumed=values)
    assert accounting["omittedFields"] == {"accounting": "record-size-limit"}
    assert not accounting["complete"] and accounting["fields"] == {}
    preview = {"module": "modules.Image", "action": "Preview", "params": {"export": {"value": "0"}}}
    record = capture_generation_inputs("preview", preview, {"prompt": "p" * 16384})
    record["fieldAccounting"] = capture_field_accounting(preview, {"export": "0"}, consumed={"export": "0"})
    for index in range(4):
        preview["params"]["unknown" + str(index) + "x" * 16384] = {"value": "private-unknown-value"}
    bound = bind_generation_input_origins("preview", preview, record)
    assert bound["fieldAccounting"]["omittedFields"] == {"accounting": "record-size-limit"}
    assert not bound["fieldAccounting"]["complete"]
    assert bound["fields"] == record["fields"]
    assert "private-unknown-value" not in json.dumps(bound)


def test_v1_receipt_remains_unchanged_and_cached_accounting_retains_effective_values(monkeypatch):
    from modiff.execution_input_provenance import apply_resolved_execution_inputs

    receipt, calls, instances, _ = run_ddim(monkeypatch)
    consumer = instances["node-04"]
    expected = deepcopy(consumer._execution_input_record)
    consumer(**consumer.params)
    assert consumer._cache_reason == "unchanged_inputs"
    assert consumer._execution_input_record == expected
    assert calls["generate"]["num_inference_steps"] == 50
    records = {
        row["nodeId"]: {key: value for key, value in row.items() if key != "fieldAccounting"}
        for row in receipt["nodes"]
    }
    old = build_resolved_execution_inputs(
        ddim_graph(), records, task_id="protocol-only-ddim", attempt_index=0, node_id="node-05"
    )
    for key in ("summary", "ambiguousFields", "unavailableFields", "graphTasks", "uncapturedNodeIds"):
        assert old.get(key) == receipt.get(key)
    retained = apply_resolved_execution_inputs(
        {"taskId": receipt["taskId"], "attemptIndex": 0, "nodeId": "node-05"}, receipt
    )
    assert retained["resolvedExecutionInputs"] == receipt
    receipt["nodes"][0]["fieldAccounting"]["fields"].clear()
    assert retained["resolvedExecutionInputs"]["nodes"][0]["fieldAccounting"]["fields"]


@pytest.mark.parametrize(
    "pipeline_class,optional,expected,inactive",
    [
        ("DDPMPipeline", {}, {}, {"eta", "class_label"}),
        ("DDIMPipeline", {"eta": 0.25}, {"eta": 0.25}, {"class_label"}),
        ("ConsistencyModelPipeline", {"class_labels": 145}, {"class_label": 145}, {"eta"}),
    ],
)
def test_unconditional_accounting_uses_actual_supported_call_arguments(pipeline_class, optional, expected, inactive):
    from modules.DiffusersImage.main import UnconditionalGenerate, IMAGE_PIPELINE_ADAPTERS, image_pipeline_contract
    from modules.DiffusersImage.call_inputs import record_image_call_inputs

    node = UnconditionalGenerate("protocol-only")
    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_class]
    values = dict(
        batch_size=1,
        seed=7,
        num_inference_steps=2,
        output_type="pil",
        eta=0.25,
        class_label=145,
        image_contract=image_pipeline_contract(adapter, "unconditional_image"),
    )
    call = dict(batch_size=1, num_inference_steps=2, output_type="pil", **optional)
    record_image_call_inputs(node, values, call, adapter)
    accounting = node._execution_input_record["fieldAccounting"]
    assert accounting["complete"]
    for field in inactive:
        assert accounting["fields"][field]["disposition"] == "inactive"
    for field, actual in expected.items():
        assert accounting["fields"][field]["disposition"] == "consumed"
        assert accounting["fields"][field]["value"] == actual


def test_accounting_origins_keep_connected_and_override_evidence_distinct(monkeypatch):
    _, _, instances, _ = run_ddim(monkeypatch)
    graph_node = deepcopy(ddim_graph()["nodes"]["node-04"])
    graph_node["params"]["eta"] = {"sourceId": "external", "sourceKey": "value", "value": 0}
    original = deepcopy(instances["node-04"]._execution_input_record)
    bound = bind_generation_input_origins("node-04", graph_node, original)
    assert bound["fieldAccounting"]["fields"]["eta"]["source"] == "connected"
    assert bound["fieldAccounting"]["fields"]["eta"]["sourceNodeId"] == "external"
    overridden = bind_generation_input_origins("node-04", graph_node, original, {"eta": 0})
    assert overridden["fieldAccounting"]["fields"]["eta"]["source"] == "override"
    assert "source" not in original["fieldAccounting"]["fields"]["eta"]
    rebound = bind_generation_input_origins("node-04", graph_node, bound, {"eta": 0})
    assert rebound["fieldAccounting"]["fields"]["eta"]["source"] == "override"
    assert "sourceNodeId" not in rebound["fieldAccounting"]["fields"]["eta"]


def test_optional_accounting_cannot_truncate_fitting_v1_ancestry_or_task_proof(monkeypatch):
    from modiff.execution_input_provenance import capture_generation_inputs, capture_field_accounting

    first, _, _, _ = run_ddim(monkeypatch)
    graph = ddim_graph()
    records = {row["nodeId"]: deepcopy(row) for row in first["nodes"]}
    previous = "node-05"
    for index in range(17):
        node_id = f"preview-{index}"
        definition = {
            "module": "modules.Image",
            "action": "Preview",
            "params": {"image": {"sourceId": previous, "sourceKey": "output"}, "export": {"value": "0" * 8000}},
        }
        graph["nodes"][node_id] = definition
        record = capture_generation_inputs(node_id, definition, {"width": 32})
        record["fieldAccounting"] = capture_field_accounting(
            definition, {"export": "0" * 8000}, consumed={"export": "0" * 8000}
        )
        records[node_id] = bind_generation_input_origins(node_id, definition, record)
        previous = node_id
    kwargs = dict(task_id="budget-only", attempt_index=0, node_id=previous)
    legacy_records = {
        key: {field: value for field, value in record.items() if field != "fieldAccounting"}
        for key, record in records.items()
    }
    old = build_resolved_execution_inputs(graph, legacy_records, **kwargs)
    new = build_resolved_execution_inputs(graph, records, **kwargs)
    assert new["fieldAccountingStatus"] == {"schemaVersion": 1, "truncated": True}
    assert not new["truncated"] and not old["truncated"]
    assert new.get("graphTasks") and new["summary"]
    comparable = {key: value for key, value in new.items() if key != "fieldAccountingStatus"}
    comparable["nodes"] = [
        {key: value for key, value in row.items() if key != "fieldAccounting"} for row in new["nodes"]
    ]
    assert comparable == old
    assert len(json.dumps(new["nodes"], ensure_ascii=True).encode("ascii")) < 262144
