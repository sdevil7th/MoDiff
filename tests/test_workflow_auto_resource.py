from copy import deepcopy
from types import SimpleNamespace

import pytest

from modiff import workflow_auto_resource as planner


def node(action, **params):
    return {"module": "modules.Image" if action == "Preview" else "modules.ModularDiffusers", "action": action,
            "params": {key: value if isinstance(value, dict) and "sourceId" in value else {"value": value} for key, value in params.items()}}


def graph(two=False, shared=False):
    nodes = {"load": node("ModelsLoader", repo_id={"source": "hub", "value": "test/model"}, dtype="bfloat16", device="cuda:0", offload_mode="none", auto_offload=False),
             "generate": node("Generate", pipeline={"sourceId": "load", "sourceKey": "pipeline"}, width=1024, height=1024, num_inference_steps=30),
             "preview": node("Preview", image={"sourceId": "generate", "sourceKey": "images"})}
    if two or shared:
        if two:
            nodes["load2"] = deepcopy(nodes["load"])
        nodes["generate2"] = node("Generate", pipeline={"sourceId": "load2" if two else "load", "sourceKey": "pipeline"}, width=1024, height=1024, num_inference_steps=30)
    return {"nodes": nodes, "paths": [list(nodes)]}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    profile = SimpleNamespace(model_type="TestPipeline", modes=("text_to_image",), loader_module="modules.ModularDiffusers", loader_action="ModelsLoader", execution_path="modular-diffusers")
    monkeypatch.setattr(planner, "resolve_execution_profiles_for_loader", lambda module, action, values: ((profile,), None) if action == "ModelsLoader" else ((), None))
    hardware = {"accelerator": {"freeBytes": 1000, "memoryKind": "dedicated"}, "systemMemory": {"availableBytes": 1000}, "offloadDisk": {"freeBytes": 1000}}
    candidate = {"id": "accepted-recipe", "modelRepo": "test/model", "modelType": "TestPipeline", "mode": "text_to_image", "loaderModule": profile.loader_module, "loaderAction": profile.loader_action, "executionPath": profile.execution_path,
                 "dtype": "bfloat16", "quantizationMode": "none", "offloadMode": "none", "autoOffload": False,
                 "canAutoRun": True, "proof": {"status": "declared_safe"}, "generation": {"width": 1024, "height": 1024, "steps": 30}, "requirements": {"systemRamBytes": 300, "vramBytes": 200, "diskFreeBytes": 10}}
    requests = []

    def recipe(payload, **kwargs):
        requests.append(deepcopy(payload))
        return {"candidates": [deepcopy(candidate)]}

    def plan(g):
        return planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path), plan_recipe=recipe, hardware=hardware)

    return plan, candidate, hardware, requests


def test_multiple_blocks_count_separate_model_owners_and_preserve_graph(setup):
    plan, _, _, requests = setup
    g = graph(two=True)
    before = deepcopy(g)
    result = plan(g)
    assert result["canAutoRun"] is True
    assert len(result["loaders"]) == 2
    assert len(requests) == 1  # Same recipe inspected once; owners still counted twice.
    assert result["retainedRequirements"]["systemRamBytes"] == 600
    assert result["requirements"]["systemRamBytes"] == 300
    assert result["strategy"] == "dependency_order_release_owners"
    assert result["loaders"][0]["consumers"] == ["generate"]
    assert g == before


def test_shared_loader_is_counted_once_with_both_consumers(setup):
    plan, _, _, _ = setup
    result = plan(graph(shared=True))
    assert result["canAutoRun"] is True
    assert result["requirements"]["systemRamBytes"] == 300
    assert result["loaders"][0]["consumers"] == ["generate", "generate2"]


def test_release_schedule_visits_shared_path_prefixes_once(setup):
    plan, _, hardware, _ = setup
    hardware["systemMemory"]["availableBytes"] = 450
    g = graph(two=True)
    g["nodes"]["generate2"]["params"]["image"] = {"sourceId": "generate", "sourceKey": "images"}
    g["paths"] = [["load", "generate", "preview"], ["load", "generate", "load2", "generate2"]]
    result = plan(g)
    assert result["canAutoRun"]
    schedule = result["schedule"]
    assert len(schedule["executionOrder"]) == len(g["nodes"])
    event = next(event for event in schedule["releases"] if "load" in event["ownerIds"])
    assert event["afterIndex"] < schedule["executionOrder"].index("load2")
    assert event["retainOutputs"] == {"generate": ["images"]}


@pytest.mark.parametrize("field", ["state_input__width", "iteration_input__width"])
def test_reviewed_workload_alias_is_planned_and_bound_to_dispatch(setup, field):
    plan, candidate, _, requests = setup
    g = graph()
    step = g["nodes"]["generate"]
    step["action"] = "ReviewedModularWorkflowStep"
    step["params"].pop("width")
    step["params"][field] = {"value": 4096}
    before = deepcopy(g)
    assert not plan(g)["canAutoRun"]  # The 1024-pixel recipe cannot authorize 4096.
    assert requests[-1]["form"]["width"] == 4096
    candidate["generation"]["width"] = 4096
    result = plan(g)
    assert result["canAutoRun"]
    assert result["resolvedFields"]["generate"][field] == 4096
    assert g == before


def test_connected_reviewed_alias_defers_and_captures_actual_supplier_output(setup):
    from modiff.workflow_auto_values import RUNTIME_VALUES
    plan, candidate, _, _ = setup
    g = graph()
    g["nodes"] = {"size": {"module": "modules.Image", "action": "LoadImage", "params": {}}, **g["nodes"]}
    step = g["nodes"]["generate"]
    step["action"] = "ReviewedModularWorkflowStep"
    step["params"].pop("width")
    step["params"]["state_input__width"] = {"sourceId": "size", "sourceKey": "width"}
    g["paths"] = [list(g["nodes"])]
    pending = plan(g)
    assert pending["canAutoRun"] and pending["requiresPreparation"]
    assert pending["preparationNodeIds"] == ["size"]
    candidate["generation"]["width"] = 2048
    token = RUNTIME_VALUES.set({("size", "width"): 2048})
    try:
        result = plan(g)
    finally:
        RUNTIME_VALUES.reset(token)
    assert result["canAutoRun"] and not result["requiresPreparation"]
    assert result["resolvedFields"]["generate"]["state_input__width"] == 2048


@pytest.mark.parametrize("kind", ["constant", "state"])
def test_lowered_iteration_workload_is_accounted_for_or_explicitly_blocked(setup, kind):
    plan, candidate, _, requests = setup
    g = graph()
    step = g["nodes"]["generate"]
    step["action"] = "ReviewedModularWorkflowStep"
    step["params"].pop("width")
    binding = {"kind": "constant", "value": 4096} if kind == "constant" else {
        "kind": "state", "sourcePath": ["denoise", "other"], "output": "width", "timing": "previous",
    }
    step["params"]["iteration_bindings"] = {"value": {"width": binding}}
    result = plan(g)
    assert not result["canAutoRun"]
    if kind == "constant":
        assert requests[-1]["form"]["width"] == 4096
        candidate["generation"]["width"] = 4096
        result = plan(g)
        assert result["canAutoRun"]
        assert result["resolvedFields"]["generate"]["iteration_bindings"] == {"width": binding}
    else:
        assert "iteration" in " ".join(result["issues"])


@pytest.mark.parametrize("batch", [1, 8])
def test_batch_size_requires_matching_candidate_evidence(setup, batch):
    plan, candidate, _, _ = setup
    g = graph()
    g["nodes"]["generate"]["params"]["num_images_per_prompt"] = {"value": batch}
    assert plan(g)["canAutoRun"] is (batch == 1)
    candidate["generation"]["batchSize"] = batch
    assert plan(g)["canAutoRun"]
    candidate["generation"]["batchSize"] = batch + 1
    assert not plan(g)["canAutoRun"]


def test_deferred_mixed_plan_hash_matches_the_returned_patch_set(setup):
    from modiff.workflow_auto_values import RUNTIME_VALUES
    plan, candidate, _, _ = setup
    candidate.update(offloadMode="model_cpu", autoOffload=True)
    g = graph(two=True)
    g["nodes"] = {"size": {"module": "modules.Image", "action": "LoadImage", "params": {}}, **g["nodes"]}
    g["nodes"]["generate2"]["params"]["width"] = {"sourceId": "size", "sourceKey": "width"}
    g["paths"] = [list(g["nodes"])]
    result = plan(g)
    assert result["canAutoRun"] and result["requiresPreparation"]
    assert result["patches"] == []
    assert result["plannedGraphHash"] == planner.workflow_graph_hash(g)
    token = RUNTIME_VALUES.set({("size", "width"): 1024})
    try:
        prepared = plan(g)
    finally:
        RUNTIME_VALUES.reset(token)
    assert not prepared["requiresPreparation"] and len(prepared["patches"]) == 4
    for patch in prepared["patches"]:
        g["nodes"][patch["nodeId"]]["params"][patch["field"]]["value"] = patch["value"]
    assert prepared["plannedGraphHash"] == planner.workflow_graph_hash(g)


@pytest.mark.parametrize("action,accepted", [("SeededGenerator", True), ("AttentionArguments", True), ("AddTensorNoise", False)])
def test_only_reviewed_tensor_data_actions_are_admitted_without_model_ownership(tmp_path, action, accepted):
    g = {"nodes": {"data": {"module": "modules.Tensor", "action": action, "params": {}}}, "paths": [["data"]]}
    result = planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path), hardware={})
    assert result["canAutoRun"] is accepted
    assert result["loaders"] == []


def test_shared_memory_is_not_added_as_extra_capacity(setup):
    plan, _, hardware, _ = setup
    hardware["accelerator"]["memoryKind"] = "shared"
    hardware["systemMemory"]["availableBytes"] = 900
    g = graph(two=True)
    g["nodes"]["generate2"]["params"]["other_model"] = {"sourceId": "load", "sourceKey": "pipeline"}
    result = plan(g)
    assert result["canAutoRun"] is False
    assert "needs 1000 bytes" in " ".join(result["issues"])


@pytest.mark.parametrize("edit", [lambda c: c.update(dtype="float16"), lambda c: c.update(modelRepo="another/model"), lambda c: c["proof"].update(status="planned"), lambda c: c["generation"].update(width=512), lambda c: c.update(requirements={})])
def test_unproven_substituted_or_incomplete_recipes_do_not_authorize(setup, edit):
    plan, candidate, _, _ = setup
    edit(candidate)
    result = plan(graph())
    assert result["canAutoRun"] is False
    assert result["patches"] == []


def test_only_resource_settings_are_proposed_and_bind_the_resulting_graph(setup):
    plan, candidate, _, _ = setup
    candidate.update(offloadMode="model_cpu", autoOffload=True)
    g = graph()
    result = plan(g)
    assert result["canAutoRun"] is True
    assert {patch["field"] for patch in result["patches"]} == {"offload_mode", "auto_offload"}
    assert result["plannedGraphHash"] != result["graphHash"]
    for patch in result["patches"]:
        g["nodes"][patch["nodeId"]]["params"][patch["field"]]["value"] = patch["value"]
    assert planner.workflow_graph_hash(g) == result["plannedGraphHash"]
    assert plan(g)["patches"] == []


def test_computed_workload_blocks_without_running_supplier(setup):
    plan, _, _, _ = setup
    g = graph()
    g["nodes"]["generate"]["params"]["width"] = {"sourceId": "load", "sourceKey": "computed_width"}
    result = plan(g)
    assert result["canAutoRun"] is False
    assert "generate.width is computed" in " ".join(result["issues"])


@pytest.mark.parametrize("kind", ["cycle", "missing", "uncovered"])
def test_invalid_graphs_fail_before_recipe_planning(setup, kind):
    plan, _, _, requests = setup
    g = graph()
    if kind == "cycle":
        g["nodes"]["load"]["params"]["x"] = {"sourceId": "generate", "sourceKey": "images"}
    elif kind == "missing":
        g["nodes"]["generate"]["params"]["pipeline"]["sourceId"] = "absent"
    else:
        g["paths"] = [["load"]]
    with pytest.raises(ValueError):
        plan(g)
    assert not requests


def test_data_only_graph_runs_in_auto_without_model_plans(setup):
    plan, _, _, requests = setup
    result = plan({"nodes": {"preview": node("Preview", image=[])}, "paths": [["preview"]]})
    assert result["canAutoRun"] is True
    assert not requests


def test_reviewed_text_conversion_is_not_a_model_and_resolves_bounded_workload(tmp_path):
    g = {"nodes": {"steps": {"module": "modules.Text", "action": "ProcessText", "params": {
        "pipeline_class": {"value": "BuiltinDataOperationV1"}, "operation": {"value": "data_conversion"},
        "source": {"value": "4"}, "target_type": {"value": "integer"}}}}, "paths": [["steps"]]}
    result = planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path), hardware={})
    assert result["canAutoRun"] is True
    assert result["loaders"] == []
    g["nodes"]["consumer"] = node("Generate", steps={"sourceId": "steps", "sourceKey": "output"})
    assert planner._literal(g["nodes"], "consumer", "steps") == 4


def test_shared_accelerator_uses_accessible_pool_but_one_system_budget(setup):
    plan, _, hardware, _ = setup
    hardware["accelerator"].update(memoryKind="shared", kind="cuda", freeBytes=1)
    hardware["runtime"] = {"hardware": {"devices": [{"type": "cuda", "memory_kind": "shared", "shared_memory_free": 500, "torch_vram_free": 600}]}}
    result = plan(graph())
    assert result["canAutoRun"] is True
    assert result["available"]["vramBytes"] == 500


def test_lora_shape_budget_uses_pinned_bytes_and_rejects_changed_hash(tmp_path):
    import hashlib
    import numpy as np
    from safetensors.numpy import save_file
    from modiff.auxiliary_lora import lora_resource_requirement
    path = tmp_path / "adapter.safetensors"
    save_file({"lora_A.weight": np.zeros((2, 4), dtype=np.float32), "lora_B.weight": np.zeros((8, 2), dtype=np.float32)}, str(path))
    adapter = node("Lora", model={"source": "local", "value": str(path)}, weight_name=path.name,
                   expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scale=1.0)
    result = lora_resource_requirement("adapter", adapter)
    assert result["tensorCount"] == 2
    assert result["systemRamBytes"] == result["vramBytes"] == (8 + 16) * 12 + 8 * 8 * 4
    adapter["params"]["expected_sha256"]["value"] = "0" * 64
    with pytest.raises(ValueError, match="SHA|hash|checksum"):
        lora_resource_requirement("adapter", adapter)


def test_sequential_owners_fit_by_releasing_the_completed_owner(setup):
    plan, _, hardware, _ = setup
    hardware['systemMemory']['availableBytes'] = 450
    result = plan(graph(two=True))
    assert result['canAutoRun']
    assert result['requirements']['systemRamBytes'] == 300
    assert result['retainedRequirements']['systemRamBytes'] == 600
    assert result['schedule']['releases'][0]['ownerIds'] == ['load']
    assert result['schedule']['executionOrder'].index('generate') < result['schedule']['executionOrder'].index('load2')


def test_overlapping_owners_cannot_be_released_before_shared_consumer(setup):
    plan, _, hardware, _ = setup
    hardware['systemMemory']['availableBytes'] = 450
    g = graph(two=True)
    g['nodes']['generate2']['params']['other_model'] = {'sourceId': 'load', 'sourceKey': 'pipeline'}
    result = plan(g)
    assert not result['canAutoRun']


def test_composed_image_output_is_retained_across_owner_release(setup):
    plan, _, hardware, _ = setup
    hardware['systemMemory']['availableBytes'] = 450
    g = graph(two=True)
    g['nodes']['generate2']['params']['image'] = {'sourceId': 'generate', 'sourceKey': 'images'}
    result = plan(g)
    assert result['canAutoRun']
    assert result['schedule']['releases'][0]['retainOutputs'] == {'generate': ['images']}


def test_nested_data_selection_and_conversion_resolve_without_executing_nodes(setup):
    plan, _, _, _ = setup
    g = graph()
    select = {'module': 'modules.Text', 'action': 'ProcessText', 'params': {'source': {'value': '512\n1024'}, 'index': {'value': 1}, 'operation': {'value': 'text_select'}}}
    convert = {'module': 'modules.Text', 'action': 'ProcessText', 'params': {'source': {'sourceId': 'select', 'sourceKey': 'output'}, 'operation': {'value': 'data_conversion'}, 'target_type': {'value': 'integer'}}}
    g['nodes'] = {'select': select, 'convert': convert, **g['nodes']}
    g['nodes']['generate']['params']['width'] = {'sourceId': 'convert', 'sourceKey': 'output'}
    g['paths'] = [list(g['nodes'])]
    before = deepcopy(g)
    result = plan(g)
    assert result['canAutoRun'] and not result['requiresPreparation']
    assert result['resolvedFields']['generate']['width'] == 1024
    assert g == before


def test_runtime_data_supplier_is_deferred_without_reading_files_or_old_cache(setup):
    plan, _, _, _ = setup
    g = graph()
    g['nodes'] = {'size': {'module': 'modules.Image', 'action': 'LoadImage', 'params': {'path': {'value': '/not/read/by/inspection.png'}}}, **g['nodes']}
    g['nodes']['generate']['params']['width'] = {'sourceId': 'size', 'sourceKey': 'width'}
    g['paths'] = [list(g['nodes'])]
    result = plan(g)
    assert result['canAutoRun'] and result['requiresPreparation']
    assert result['preparationNodeIds'] == ['size']
    assert result['patches'] == []
    from modiff.workflow_auto_values import RUNTIME_VALUES
    token = RUNTIME_VALUES.set({('size', 'width'): 1024})
    try:
        ready = plan(g)
    finally:
        RUNTIME_VALUES.reset(token)
    assert ready['canAutoRun'] and not ready['requiresPreparation']
    assert ready['resolvedFields']['generate']['width'] == 1024


def test_owner_release_retains_images_and_drops_model_references():
    from modiff.workflow_auto_lifecycle import release_owner_caches
    from PIL import Image
    import weakref
    class Model: pass
    model = Model()
    ref = weakref.ref(model)
    calls = []
    manager = SimpleNamespace(cache={'weight': model})
    image = Image.new('RGB', (8, 8), 'red')
    server = SimpleNamespace(node_cache={
        'load': SimpleNamespace(output={'pipeline': model}, params={}, _mm_models=['weight']),
        'generate': SimpleNamespace(output={'images': [image]}, params={'pipeline': model}, _mm_models=[]),
    }, _release_node_modular_components=lambda ids: calls.append(set(ids)),
       _best_effort_device_cache_clear=lambda: [], _best_effort_allocator_trim=lambda: (True, []))
    del model
    result = release_owner_caches(server, {'nodeIds': ['load', 'generate'], 'ownerIds': ['load'], 'retainOutputs': {'generate': ['images']}}, manager)
    assert ref() is None
    assert server.node_cache['generate'].output['images'][0].getpixel((0, 0)) == (255, 0, 0)
    assert not manager.cache and 'load' not in server.node_cache
    assert result['retainedOutputNodes'] == ['generate']


def test_opaque_output_blocks_release_atomically():
    from modiff.workflow_auto_lifecycle import release_owner_caches
    model = object()
    cached = SimpleNamespace(output={'pipeline': model}, params={}, _mm_models=[])
    server = SimpleNamespace(node_cache={'load': cached})
    with pytest.raises(ValueError, match='opaque'):
        release_owner_caches(server, {'nodeIds': ['load'], 'ownerIds': ['load'], 'retainOutputs': {'load': ['pipeline']}}, SimpleNamespace(cache={}))
    assert server.node_cache['load'] is cached


def test_stage_gate_uses_actual_free_memory_without_reclaimable_estimates():
    from modiff.workflow_auto_lifecycle import assert_next_owner_capacity
    owner = {'ownerId': 'second', 'requirements': {'systemRamBytes': 300, 'vramBytes': 200}}
    hardware = {'systemMemory': {'availableBytes': 490}, 'accelerator': {'memoryKind': 'shared', 'freeBytes': 400}}
    with pytest.raises(ValueError, match='actual free memory'):
        assert_next_owner_capacity(None, owner, hardware)
    hardware['systemMemory']['availableBytes'] = 600
    assert_next_owner_capacity(None, owner, hardware)


def test_data_preparation_runs_supplier_once_and_never_uses_prior_output():
    from modiff.server import WebServer
    from modiff.workflow_auto_values import RUNTIME_VALUES
    calls = []
    supplier = SimpleNamespace(output={'width': 1}, invalidate_cache=lambda: calls.append('invalidate'))
    server = SimpleNamespace(node_cache={'source': supplier}, interrupt_flag=False)
    def execute(node_id, node, sid):
        calls.append(node_id)
        supplier.output = {'width': 1024}
    server.execute_node = execute
    server._auto_resource_contract_error = ValueError
    server._build_workflow_auto_plan = lambda g: {'canAutoRun': True, 'requiresPreparation': False, 'preparationNodeIds': [], 'seen': RUNTIME_VALUES.get()[('source', 'width')]}
    g = {'sid': 'session', 'nodes': {'source': {'module': 'modules.Image', 'action': 'Load', 'params': {}}}, 'paths': [['source'], ['source']]}
    plan, prepared = WebServer._prepare_workflow_auto_data(server, g, {'requiresPreparation': True, 'preparationNodeIds': ['source']})
    assert calls == ['invalidate', 'source'] and plan['seen'] == 1024 and prepared == {'source'}
    assert not RUNTIME_VALUES.get()


def test_existing_executor_releases_before_next_loader_and_retains_downstream_image(setup, monkeypatch, tmp_path):
    from modiff.server import WebServer
    import modiff.workflow_auto_lifecycle as lifecycle
    from PIL import Image
    import weakref
    plan, _, hardware, _ = setup
    hardware['systemMemory']['availableBytes'] = 450
    graph_data = graph(two=True)
    graph_data['nodes']['generate2']['params']['image'] = {'sourceId': 'generate', 'sourceKey': 'images'}
    graph_data['sid'] = 'test'
    graph_data['runtimeHints'] = {'resourceMode': 'auto', 'workflowAutoPlan': {'schemaVersion': 1, 'graphHash': planner.workflow_graph_hash(graph_data)}}
    app = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    app.current_task = {'task_id': 'lifetime-test', 'progress': 0}
    app.interrupt_flag = False
    app._build_workflow_auto_plan = plan
    app._runtime_gpu_cleanup_idle = lambda: app.node_cache.clear()
    app._prepare_auto_runtime_for_graph = lambda _: None
    app._runtime_fingerprint = lambda: {'fingerprint': 'unit'}
    app._runtime_measurement = lambda **_: {'elapsedSeconds': 0}
    app._record_auto_resource_success = lambda *a, **k: None
    app._record_optimization_observations = lambda *a, **k: []
    app._release_node_modular_components = lambda _: 0
    app._best_effort_device_cache_clear = lambda: []
    app._best_effort_allocator_trim = lambda: (True, [])
    monkeypatch.setattr(lifecycle, 'assert_next_owner_capacity', lambda *args: None)
    class Model: pass
    calls, refs, messages = [], {}, []
    app.queue_message = messages.append
    def execute(node_id, node, sid):
        calls.append(node_id)
        if node_id == 'load2':
            assert refs['load']() is None
            assert app.node_cache['generate'].output['images'][0].getpixel((0, 0)) == (255, 0, 0)
        if node_id.startswith('load'):
            model = Model()
            refs[node_id] = weakref.ref(model)
            app.node_cache[node_id] = SimpleNamespace(output={'pipeline': model}, params={}, _mm_models=[])
        elif node_id.startswith('generate'):
            model = app.node_cache[node['params']['pipeline']['sourceId']].output['pipeline']
            app.node_cache[node_id] = SimpleNamespace(output={'images': [Image.new('RGB', (8, 8), 'red')]}, params={'pipeline': model}, _mm_models=[])
        else:
            app.node_cache[node_id] = SimpleNamespace(output={}, params={}, _mm_models=[])
    app.execute_node = execute
    for _ in range(2):
        app._execute_graph(deepcopy(graph_data))
    assert calls.count('load') == calls.count('load2') == 2
    completed = [item for item in messages if item.get('type') == 'graph_completed']
    assert len(completed) == 2
    assert completed[0]['runtimePreparation']['workflowAuto']['releases'][0]['ownerIds'] == ['load']


@pytest.mark.parametrize('shared_outside', [False, True])
@pytest.mark.parametrize('two', [False, True])
def test_computed_run_applies_only_representable_offload_updates_before_loading(setup, tmp_path, shared_outside, two):
    from modiff.server import WebServer
    plan, candidate, _, _ = setup
    candidate.update(offloadMode='model_cpu', autoOffload=True)
    g = graph(two=two)
    if two:
        # Overlap owners to test the deferred submission independently of the
        # separately tested cache-release schedule and hardware probes.
        g['nodes']['generate2']['params']['other_model'] = {'sourceId': 'load', 'sourceKey': 'pipeline'}
    g['nodes'] = {'size': {'module': 'modules.Image', 'action': 'Load', 'params': {}}, **g['nodes']}
    g['nodes']['generate']['params']['width'] = {'sourceId': 'size', 'sourceKey': 'width'}
    g['paths'] = [list(g['nodes'])]
    g['sid'] = 'test'
    receipt = {'schemaVersion': 1, 'graphHash': plan(g)['plannedGraphHash']}
    if shared_outside:
        receipt['resourceControlGroups'] = [[{'nodeId': 'load', 'field': 'offload_mode'}, {'nodeId': 'outside-scope', 'field': 'offload_mode'}]]
    g['runtimeHints'] = {'resourceMode': 'auto', 'workflowAutoPlan': receipt}
    app = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    app.current_task = {'task_id': 'computed-test', 'progress': 0}
    app.interrupt_flag = False
    app._build_workflow_auto_plan = plan
    app._prepare_auto_runtime_for_graph = lambda _: None
    app._runtime_fingerprint = lambda: {'fingerprint': 'unit'}
    app._runtime_measurement = lambda **_: {'elapsedSeconds': 0}
    app._record_auto_resource_success = lambda *a, **k: None
    app._record_optimization_observations = lambda *a, **k: []
    calls, messages = [], []
    app.queue_message = messages.append
    def execute(node_id, node, sid):
        calls.append(node_id)
        if node_id == 'load':
            assert node['params']['offload_mode']['value'] == 'model_cpu'
            assert app._workflow_auto_resolved_fields['load']['offload_mode'] == 'model_cpu'
        app.node_cache[node_id] = SimpleNamespace(output={'width': 1024}, params={}, _mm_models=[])
    app.execute_node = execute
    if shared_outside:
        with pytest.raises(RuntimeError, match='shared control outside'):
            app._execute_graph(g)
        assert calls == ['size']
    else:
        app._execute_graph(g)
        assert calls.count('size') == 1 and calls.index('size') < calls.index('load')
        changed = next(item for item in messages if item['type'] == 'auto_resource_plan_applied')
        assert {item['field'] for item in changed['resourceUpdates']} == {'offload_mode', 'auto_offload'}
        assert g['runtimeHints']['workflowAutoPlan']['graphHash'] == planner.workflow_graph_hash(g)


def test_release_preserves_memory_manager_ids_shared_by_surviving_nodes():
    from modiff.workflow_auto_lifecycle import release_owner_caches
    manager = SimpleNamespace(cache={'shared': object(), 'expired': object()})
    app = SimpleNamespace(
        node_cache={'expired': SimpleNamespace(output={}, _mm_models=['shared', 'expired']),
                    'live': SimpleNamespace(output={}, _mm_models=['shared'])},
        _release_node_modular_components=lambda _: 0,
        _best_effort_device_cache_clear=lambda: [],
        _best_effort_allocator_trim=lambda: (True, []),
    )
    release_owner_caches(app, {'nodeIds': ['expired'], 'ownerIds': ['expired'], 'retainOutputs': {}}, manager)
    assert set(manager.cache) == {'shared'}
    assert app.node_cache['live']._mm_models == ['shared']


def test_release_rejects_a_model_reference_that_survives_ownership_cleanup(monkeypatch):
    import sys
    from modiff.workflow_auto_lifecycle import release_owner_caches
    class Model:
        def parameters(self): return ()
    held = Model()
    components = SimpleNamespace(collections={'expired': {'model'}}, components={'model': held})
    monkeypatch.setitem(sys.modules, 'modules.ModularDiffusers', SimpleNamespace(components=components))
    def prune(_):
        components.collections.clear()
        components.components.clear()
        return 1
    app = SimpleNamespace(node_cache={}, _release_node_modular_components=prune,
                          _best_effort_device_cache_clear=lambda: [],
                          _best_effort_allocator_trim=lambda: (True, []))
    with pytest.raises(ValueError, match='model references remain alive'):
        release_owner_caches(app, {'nodeIds': ['expired'], 'ownerIds': ['expired'], 'retainOutputs': {}}, SimpleNamespace(cache={}))
