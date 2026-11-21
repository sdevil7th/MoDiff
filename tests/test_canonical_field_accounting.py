"""No-weight receipt protocol checks using the unchanged normal DDIM export.

The fixture is the normal frontend export of the checked-in canonical creator,
not a substitute workflow. Library allocation/inference are mocked; these tests
are not runtime, hardware, output quality or release qualification.
"""

import hashlib
import json
from copy import deepcopy
from functools import wraps
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


def test_empty_execution_profile_is_accounted_without_granting_a_stage_route(monkeypatch):
    graph = ddim_graph()
    graph['nodes']['node-01']['params']['execution_profile_id'] = {'value': ''}
    receipt, _, _, _ = run_ddim(monkeypatch, graph)
    loader = next(row for row in receipt['nodes'] if row['action'] == 'LoadPipeline')
    accounting = loader['fieldAccounting']
    assert accounting['complete'], accounting
    assert accounting['fields']['execution_profile_id'] == {
        'disposition': 'inactive', 'inputValue': '', 'value': '',
        'source': 'literal', 'reason': 'adapter-inactive',
    }


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


def run_conditional_image(monkeypatch, pipeline_class, **changes):
    """Execute the built-in node with the installed upstream call signature."""
    from inspect import signature

    from modules.DiffusersImage import main as image

    calls = {}
    adapter = image.IMAGE_PIPELINE_ADAPTERS[pipeline_class]
    upstream = image.pipeline_class_from_name(adapter.load_pipeline_class).__call__

    @wraps(upstream)
    def generate(self, **kwargs):
        signature(upstream).bind(self, **kwargs)
        calls.update(kwargs)
        return SimpleNamespace(images=[Image.new('RGB', (kwargs['width'], kwargs['height']))])

    kernel = type(pipeline_class, (), {'__call__': generate, '_execution_device': 'cpu'})()
    node = image.Generate('generate')
    node.progress = lambda *a, **kw: None
    requested = dict(image_contract=image.image_pipeline_contract(adapter, 'text_to_image'),
                     prompt='Two silver cups', negative_prompt='', width='512', height=512,
                     seed=42, num_inference_steps=4, guidance_scale=5, max_sequence_length=224,
                     layers=4, resolution=640, cfg_normalize=False, use_en_prompt=False,
                     pag_scale=2.5, pag_adaptive_scale=0.5, strength=0.8, padding_mask_crop=0,
                     output_type='pil')
    requested.update(changes)
    output = node(pipeline=kernel, **requested)
    assert output['images'][0].size == (512, 512)
    definition = {'module': 'modules.DiffusersImage', 'action': 'Generate', 'params': {
        key: {'value': value} for key, value in requested.items()
    }}
    record = bind_generation_input_origins(node.node_id, definition, node._execution_input_record)
    receipt = build_resolved_execution_inputs({'nodes': {node.node_id: definition}}, {node.node_id: record},
        task_id='protocol-only-conditional', attempt_index=0, node_id=node.node_id)
    return receipt, calls, requested


@pytest.mark.parametrize('pipeline_class,pag,sequence_parameter', [
    ('StableDiffusionPipeline', False, None),
    ('StableDiffusionPAGPipeline', True, None),
    ('PRXPipeline', False, 'tokenizer_max_length'),
])
def test_conditional_image_accounting_matches_actual_call(monkeypatch, pipeline_class, pag, sequence_parameter):
    receipt, calls, requested = run_conditional_image(monkeypatch, pipeline_class)
    record = receipt['nodes'][0]
    accounting = record['fieldAccounting']
    assert accounting['complete'], accounting
    fields = accounting['fields']
    assert fields.keys() == requested.keys()
    assert fields['image_contract']['disposition'] == 'metadata'
    for name in ('prompt', 'negative_prompt', 'width', 'height', 'num_inference_steps', 'guidance_scale', 'output_type'):
        assert fields[name] == {'source': 'literal', 'disposition': 'consumed',
                                'inputValue': requested[name], 'value': calls[name]}
    assert fields['seed']['value'] == calls['generator'].initial_seed() == 42
    for name in ('pag_scale', 'pag_adaptive_scale'):
        assert (name in calls) == pag
        assert fields[name]['disposition'] == ('consumed' if pag else 'inactive')
        assert fields[name]['value'] == (calls[name] if pag else requested[name])
        assert name not in record['fields']  # No legacy receipt/summary expansion.
    sequence = fields['max_sequence_length']
    assert sequence['disposition'] == ('consumed' if sequence_parameter else 'inactive')
    if sequence_parameter:
        assert sequence['value'] == calls[sequence_parameter] == 224
    assert fields['layers']['disposition'] == 'inactive'
    assert fields['layers']['inputValue'] == fields['layers']['value'] == requested['layers'] == 4


@pytest.mark.parametrize('reviewed', [True, False])
@pytest.mark.parametrize('pipeline_name,width,steps,guidance,strength,sequence_limit,active_controls,resize', [
    ('StableDiffusionImg2ImgPipeline', 512, 30, 7.5, 0.8, 512,
     {'negative_prompt', 'guidance_scale', 'strength'}, (48, 32)),
    ('StableDiffusionXLImg2ImgPipeline', 1024, 30, 5.0, 0.8, 512,
     {'negative_prompt', 'guidance_scale', 'strength'}, None),
    ('LatentConsistencyModelImg2ImgPipeline', 1024, 4, 8.5, 0.8, 256,
     {'guidance_scale', 'strength'}, None),
    ('DreamLiteMobilePipeline', 1024, 4, 0.0, 0.8, 200,
     {'width', 'height', 'max_sequence_length'}, None),
    ('SanaSprintImg2ImgPipeline', 1024, 2, 4.5, 0.5, 300,
     {'width', 'height', 'max_sequence_length', 'guidance_scale', 'strength'}, None),
    ('SanaSprintImg2ImgPipeline', 1024, 4, 4.5, 0.5, 300,
     {'width', 'height', 'max_sequence_length', 'guidance_scale', 'strength'}, None),
])
def test_direct_image_edit_accounts_for_the_real_normalized_argument(
        monkeypatch, tmp_path, reviewed, pipeline_name, width, steps, guidance, strength, sequence_limit,
        active_controls, resize):
    from inspect import signature

    from modiff.model_artifact_catalog import require_catalog_revision
    from modules.DiffusersImage import main as image
    from modules.Image import main as media

    adapter = image.get_image_pipeline_adapter(pipeline_name)
    upstream = image.pipeline_class_from_name(pipeline_name).__call__
    source_size = ((1024, 1024) if pipeline_name == 'StableDiffusionXLImg2ImgPipeline' else
                   (64, 48) if resize else (512, 512))
    source = Image.new('RGB', source_size, (4, 90, 160))
    if resize:
        normalized = source.resize(resize)
        monkeypatch.setattr(image, 'prepare_reference_images', lambda _value, _adapter: normalized)
    else:
        fixture = tmp_path / 'source.png'
        source.save(fixture)
        monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
        monkeypatch.setattr(media, 'resolve_runtime_input_path', lambda _path: fixture)
        load = media.Load('source-image')
        source = normalized = load(file=['source.png'], alpha_channel='ignore')['image']
    calls = {}

    @wraps(upstream)
    def edit(self, **kwargs):
        signature(upstream).bind(self, **kwargs)
        calls.update(kwargs)
        return SimpleNamespace(images=[kwargs['image']])

    pipeline = type(pipeline_name, (), {'__call__': edit, '_execution_device': 'cpu'})()
    revision = require_catalog_revision(adapter.default_repo) if reviewed else None
    source_repo = adapter.default_repo if reviewed else 'local-model'
    source_kind = 'hub' if reviewed else 'local'
    if not reviewed and pipeline_name == 'StableDiffusionXLImg2ImgPipeline':
        # An exact repository name by itself is insufficient.
        revision, source_repo, source_kind = '0' * 40, adapter.default_repo, 'hub'
    image._tag_image_pipeline(pipeline, adapter, 'edit_image', source_repo, source_kind, revision)
    node = image.Edit('edit')
    node.progress = lambda *a, **kw: None
    requested = dict(image_contract=image.image_pipeline_contract(adapter, 'edit_image'),
                     prompt='Flat travel poster', negative_prompt='', width=str(width), height=width, seed=42,
                     num_inference_steps=steps, guidance_scale=guidance, strength=strength, reference_strength=1.0,
                     max_sequence_length=sequence_limit, output_type='pil', cfg_normalize=False,
                     control_guidance_start=0, control_guidance_end=1, image_guidance_scale=1.5,
                     layers=4, padding_mask_crop=0, pag_scale=3, pag_adaptive_scale=0,
                     resolution=640, use_en_prompt=False)
    if not reviewed and pipeline_name == 'StableDiffusionXLImg2ImgPipeline':
        with pytest.raises(ValueError, match='does not match the reviewed pin'):
            node(pipeline=pipeline, image=source, **requested)
        assert calls == {}
        return
    output = node(pipeline=pipeline, image=source, **requested)
    assert output['images'] == [normalized]
    assert calls['image'] is normalized
    active = {'prompt', 'num_inference_steps', 'output_type'} | active_controls
    assert calls.keys() & requested.keys() == active
    definition = {'module': 'modules.DiffusersImage', 'action': 'Edit', 'params': {
        **{key: {'value': value} for key, value in requested.items()},
        'image': {'sourceId': 'source-image', 'sourceKey': 'image'},
        'pipeline': {'sourceId': 'model', 'sourceKey': 'pipeline'},
    }}
    original = node._execution_input_record
    bound = bind_generation_input_origins(node.node_id, definition, original)
    assert bound['fields']['executed_task']['value'] == 'edit_image'
    if pipeline_name == 'SanaSprintImg2ImgPipeline' and steps != 2:
        assert calls['intermediate_timesteps'] is None
        assert 'fieldAccounting' not in bound
        return
    if not reviewed:
        assert 'fieldAccounting' not in bound
        return

    accounting = bound['fieldAccounting']
    assert accounting['complete'], accounting
    assert accounting['fields'].keys() == requested.keys()
    assert accounting['fields']['image_contract']['disposition'] == 'metadata'
    for key in active:
        assert accounting['fields'][key] == {
            'source': 'literal', 'disposition': 'consumed', 'inputValue': requested[key], 'value': calls[key],
        }
    assert accounting['fields']['seed']['value'] == calls['generator'].initial_seed() == 42
    for key in requested.keys() - active - {'seed', 'image_contract'}:
        assert accounting['fields'][key] == {
            'source': 'literal', 'disposition': 'inactive', 'reason': 'adapter-inactive',
            'inputValue': requested[key], 'value': requested[key],
        }
    projection = accounting['inputProjections']['image']
    assert projection['kind'] == 'decoded-image-v1'
    assert projection['values'] == {
        'mode': 'RGB', 'width': normalized.width, 'height': normalized.height,
        'byteSize': normalized.width * normalized.height * 3,
        'sha256': hashlib.sha256(normalized.tobytes()).hexdigest(),
    }
    if resize:
        assert projection['values']['sha256'] != hashlib.sha256(source.tobytes()).hexdigest()
    else:
        media_evidence = load._execution_input_record['mediaEvidence']
        decoded = media_evidence['outputs']['image']
        assert decoded == [{'kind': 'image', **projection['values']}]
        assert media_evidence['files']['file'][0]['sha256'] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert (projection['source'], projection['sourceNodeId'], projection['sourcePortId']) == (
        'connected', 'source-image', 'image',
    )
    overridden = bind_generation_input_origins(node.node_id, definition, original, {'image': source})
    assert not overridden['fieldAccounting']['complete']
    assert overridden['fieldAccounting']['inputProjections']['image']['source'] == 'override'


def test_conditional_accounting_holds_stale_metadata_and_unreviewed_literals(monkeypatch):
    receipt, _, _ = run_conditional_image(monkeypatch, 'StableDiffusionPipeline',
        image_contract={'pipelineClass': 'ObsoletePipeline'}, unreviewed_control='untrusted-value')
    accounting = receipt['nodes'][0]['fieldAccounting']
    assert not accounting['complete']
    assert accounting['omittedFields'] == {'image_contract': 'metadata-contract-mismatch'}
    assert 'unreviewed_control' in accounting['unaccountedFields']
    assert 'untrusted-value' not in json.dumps(receipt)


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
