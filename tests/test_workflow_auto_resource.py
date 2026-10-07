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
    profile = SimpleNamespace(id="test:modular", model_type="TestPipeline", modes=("text_to_image",), loader_module="modules.ModularDiffusers", loader_action="ModelsLoader", execution_path="modular-diffusers")
    monkeypatch.setattr(planner, "resolve_execution_profiles_for_loader", lambda module, action, values: ((profile,), None) if action == "ModelsLoader" else ((), None))
    hardware = {"accelerator": {"freeBytes": 1000, "memoryKind": "dedicated"}, "systemMemory": {"availableBytes": 1000}, "offloadDisk": {"freeBytes": 1000}}
    candidate = {"id": "accepted-recipe", "executionProfileId": profile.id, "modelRepo": "test/model", "modelType": "TestPipeline", "mode": "text_to_image", "loaderModule": profile.loader_module, "loaderAction": profile.loader_action, "executionPath": profile.execution_path,
                 "dtype": "bfloat16", "quantizationMode": "none", "offloadMode": "none", "autoOffload": False,
                 "canAutoRun": True, "proof": {"status": "declared_safe"}, "generation": {"width": 1024, "height": 1024, "steps": 30}, "requirements": {"systemRamBytes": 300, "vramBytes": 200, "diskFreeBytes": 10}}
    requests = []

    def recipe(payload, **kwargs):
        requests.append(deepcopy(payload))
        return {"candidates": [deepcopy(candidate)]}

    def plan(g, **kwargs):
        kwargs.pop('dispatch', None)
        return planner.build_workflow_auto_plan(g, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path), plan_recipe=recipe, hardware=hardware, **kwargs)

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
    assert result["requirements"]["systemRamBytes"] == 600
    assert result["strategy"] == "dependency_order_retained_owners"
    assert result["schedule"] is None
    assert result["loaders"][0]["consumers"] == ["generate"]
    assert g == before


def test_shared_loader_is_counted_once_with_both_consumers(setup):
    plan, _, _, _ = setup
    result = plan(graph(shared=True))
    assert result["canAutoRun"] is True
    assert result["requirements"]["systemRamBytes"] == 300
    assert result["loaders"][0]["consumers"] == ["generate", "generate2"]


def test_machine_ram_capacity_is_not_an_additional_working_budget(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate["requirements"] = {"memorySemantics": "machine_capacity", "minimum": {
        "systemRamBytes": 8 * gib, "vramBytes": 0, "diskFreeBytes": 0}}
    hardware["systemMemory"].update(totalBytes=32 * gib, availableBytes=4279840768)
    hardware["accelerator"].update(totalBytes=16 * gib, freeBytes=12 * gib)
    result = plan(graph())
    assert result["capacityRequirements"]["systemRamBytes"] == 8 * gib
    assert result["requirements"]["systemRamBytes"] == 4 * gib
    assert result["workingMemoryPolicy"] == "runtime_headroom_policy"
    assert not result["canAutoRun"]  # Actual free RAM remains below the existing safety floor.
    assert "needs 4.00 GiB" in " ".join(result["issues"])
    assert "needs 8.00 GiB" not in " ".join(result["issues"])


def test_machine_capacity_classes_take_maximum_across_independent_owners(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate["requirements"] = {"memorySemantics": "machine_capacity", "minimum": {
        "systemRamBytes": 8 * gib, "vramBytes": 8 * gib, "diskFreeBytes": 10}}
    hardware["systemMemory"].update(totalBytes=32 * gib, availableBytes=5 * gib)
    hardware["accelerator"].update(totalBytes=16 * gib, freeBytes=4 * gib)
    result = plan(graph(two=True))
    assert result["canAutoRun"]
    assert result["capacityRequirements"] == {"systemRamBytes": 8 * gib, "vramBytes": 8 * gib}
    assert result["requirements"] == {"systemRamBytes": 4 * gib, "vramBytes": 2 * gib, "diskFreeBytes": 20}
    assert all(owner["workingMemoryPolicy"] == "runtime_headroom_policy" for owner in result["loaders"])


def test_explicit_working_demands_remain_additive_and_owner_creditable(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate["requirements"] = {"memorySemantics": "machine_capacity", "minimum": {
        "systemRamBytes": 8 * gib, "vramBytes": 8 * gib, "diskFreeBytes": 0}}
    candidate["workingMemoryRequirements"] = {"systemRamBytes": 6 * gib, "vramBytes": 4 * gib}
    hardware["systemMemory"].update(totalBytes=32 * gib, availableBytes=7 * gib)
    hardware["accelerator"].update(totalBytes=16 * gib, freeBytes=7 * gib)
    g = graph(two=True)
    # Both models are simultaneously needed; an early release cannot hide their working demand.
    g["nodes"]["generate2"]["params"]["other_model"] = {"sourceId": "load", "sourceKey": "pipeline"}
    result = plan(g)
    assert not result["canAutoRun"]
    assert result["requirements"]["systemRamBytes"] == 12 * gib
    assert result["requirements"]["vramBytes"] == 8 * gib
    assert result["workingMemoryPolicy"] == "explicit_working_demand"


def test_machine_capacity_still_rejects_a_smaller_machine(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate["requirements"] = {"memorySemantics": "machine_capacity", "minimum": {
        "systemRamBytes": 32 * gib, "vramBytes": 0, "diskFreeBytes": 0}}
    hardware["systemMemory"].update(totalBytes=16 * gib, availableBytes=12 * gib)
    result = plan(graph())
    assert not result["canAutoRun"]
    assert "32 GiB total" in " ".join(result["issues"])


def test_capacity_policy_shared_pool_and_disk_use_actual_free_memory(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate["requirements"] = {"memorySemantics": "machine_capacity", "minimum": {
        "systemRamBytes": 8 * gib, "vramBytes": 8 * gib, "diskFreeBytes": 2000}}
    hardware["systemMemory"].update(totalBytes=32 * gib, availableBytes=5 * gib)
    hardware["accelerator"].update(totalBytes=16 * gib, freeBytes=5 * gib, memoryKind="shared")
    result = plan(graph())
    assert not result["canAutoRun"]
    assert "needs 6.00 GiB" in " ".join(result["issues"])
    assert "diskFreeBytes" in " ".join(result["issues"])


def test_offloaded_machine_capacity_uses_shared_accessible_gpu_pool(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate.update(offloadMode="model_cpu", autoOffload=True)
    candidate["requirements"] = {"memorySemantics": "machine_capacity", "minimum": {
        "accelerator": "cuda", "systemRamBytes": 32 * gib, "vramBytes": 24 * gib}}
    hardware["systemMemory"].update(totalBytes=121 * gib, availableBytes=60 * gib)
    hardware["accelerator"].update(kind="cuda", totalBytes=2 * gib, freeBytes=1 * gib,
                                   memoryKind="shared", accessibleTotalBytes=96 * gib, sharedTotalBytes=121 * gib)
    hardware["runtime"] = {"hardware": {"devices": [{"type": "cuda", "memory_kind": "shared",
        "shared_memory_free": 60 * gib, "torch_vram_free": 90 * gib}]}}
    result = plan(graph())
    assert result["canAutoRun"]
    assert result["capacityRequirements"]["vramBytes"] == 24 * gib
    assert result["available"]["vramBytes"] == 60 * gib  # One host pool, not extra free VRAM.
    assert result["requirements"]["vramBytes"] == 2 * gib
    candidate.update(offloadMode="none", autoOffload=False)
    resident = plan(graph())
    assert not resident["canAutoRun"]  # Preserve existing direct-residency capacity ranking.
    assert "24 GiB total" in " ".join(resident["issues"])


def recipe_graph(offload="none", **recipe_values):
    g = graph()
    g['nodes'] = {
        'quant': {'module': 'modules.DiffusersRuntime', 'action': 'PipelineQuantizationConfigV2',
                  'params': {'backend': {'value': 'none'}, 'component_overrides': {'value': '{}'}}},
        'recipe': {'module': 'modules.DiffusersRuntime', 'action': 'DiffusersExecutionRecipe',
                   'params': {'offload_mode': {'value': offload}, 'device': {'value': 'cuda:0'},
                              'quantization_config': {'sourceId': 'quant', 'sourceKey': 'quantization_config'},
                              **{key: {'value': value} for key, value in recipe_values.items()}}},
        **g['nodes'],
    }
    g['nodes']['load']['params']['execution_recipe'] = {'sourceId': 'recipe', 'sourceKey': 'execution_recipe'}
    g['paths'] = [list(g['nodes'])]
    return g


def test_standard_recipe_helpers_are_owned_data_and_actual_offload_is_authoritative(setup):
    plan, candidate, _, requests = setup
    candidate.update(offloadMode='model_cpu', autoOffload=True)
    g = recipe_graph(offload='model_cpu', attention_backend='_native_math')
    before = deepcopy(g)
    result = plan(g)
    assert result['canAutoRun']
    assert requests[-1]['form']['offloadMode'] == 'model_cpu'
    assert not any('no connected' in issue for issue in result['issues'])
    assert {patch['field'] for patch in result['patches']} == {'offload_mode', 'auto_offload'}
    assert all(patch['nodeId'] == 'load' for patch in result['patches'])
    assert g == before


def test_auto_patches_authoritative_recipe_and_loader_mirror_together(setup):
    plan, candidate, _, _ = setup
    candidate.update(offloadMode='model_cpu', autoOffload=True)
    result = plan(recipe_graph())
    assert result['canAutoRun']
    assert {'nodeId': 'recipe', 'field': 'offload_mode', 'value': 'model_cpu'} in result['patches']
    assert {'nodeId': 'load', 'field': 'offload_mode', 'value': 'model_cpu'} in result['patches']
    assert {'nodeId': 'load', 'field': 'auto_offload', 'value': True} in result['patches']


@pytest.mark.parametrize('override', [
    {'regional_compile': True}, {'layerwise_casting': True}, {'channels_last': True},
    {'device_map': 'balanced'}, {'denoiser_cache': 'first_block'}, {'max_memory': '{"0":"1GiB"}'},
    {'vae_tiling': False}, {'attention_backend': 'flash'},
])
def test_recipe_resource_overrides_are_never_ignored_by_auto(setup, override):
    plan, _, _, _ = setup
    result = plan(recipe_graph(**override))
    assert not result['canAutoRun']
    assert result['patches'] == []
    assert 'Custom memory' in ' '.join(result['issues'])


@pytest.mark.parametrize('backend,overrides', [('quanto_int8', '{}'), ('none', '{"transformer":"bnb_4bit"}')])
def test_connected_active_per_component_quantizer_needs_its_own_recipe(setup, backend, overrides):
    plan, _, _, _ = setup
    g = recipe_graph()
    g['nodes']['quant']['params'].update(backend={'value': backend}, component_overrides={'value': overrides})
    result = plan(g)
    assert not result['canAutoRun']
    assert 'per-component quantization' in ' '.join(result['issues'])


def test_connected_recipe_offload_control_cannot_be_silently_overwritten(setup):
    plan, candidate, _, _ = setup
    candidate.update(offloadMode='model_cpu', autoOffload=True)
    g = recipe_graph()
    g['nodes'] = {'control': {'module': 'modules.Primitive', 'action': 'String', 'params': {'value': {'value': 'none'}}}, **g['nodes']}
    g['nodes']['recipe']['params']['offload_mode'] = {'sourceId': 'control', 'sourceKey': 'value'}
    g['paths'] = [list(g['nodes'])]
    result = plan(g)
    assert not result['canAutoRun']
    assert 'Connected recipe.offload_mode' in ' '.join(result['issues'])
    assert result['patches'] == []


def test_unknown_recipe_supplier_stays_unreviewed(setup):
    plan, _, _, _ = setup
    g = recipe_graph()
    g['nodes']['recipe']['action'] = 'UnknownRecipe'
    result = plan(g)
    assert not result['canAutoRun']
    assert 'cannot inspect this execution recipe supplier' in ' '.join(result['issues'])


def test_outpaint_geometry_is_deferred_to_existing_data_preparation(setup):
    plan, _, _, _ = setup
    g = graph()
    g['nodes'] = {'source': {'module': 'modules.Image', 'action': 'LoadImage', 'params': {}},
                  'canvas': {'module': 'modules.DiffusersImage', 'action': 'OutpaintCanvas',
                             'params': {'image': {'sourceId': 'source', 'sourceKey': 'image'}, 'width': {'value': 1024}}}, **g['nodes']}
    g['nodes']['generate']['params']['width'] = {'sourceId': 'canvas', 'sourceKey': 'width_out'}
    g['paths'] = [list(g['nodes'])]
    result = plan(g)
    assert result['canAutoRun'] and result['requiresPreparation']
    assert result['preparationNodeIds'] == ['canvas', 'source']


def test_native_runtime_policy_is_preserved_and_connected_controls_are_bound(setup):
    plan, _, _, _ = setup
    g = graph()
    g['nodes'] = {'policy': {'module': 'modules.Primitive', 'action': 'String',
                            'params': {'value': {'value': '_native_math'}}}, **g['nodes']}
    g['nodes']['load']['params'].update(
        attention_backend={'sourceId': 'policy', 'sourceKey': 'value'},
        vae_slicing={'value': True}, vae_tiling={'value': True},
    )
    g['paths'] = [list(g['nodes'])]
    before = deepcopy(g)
    result = plan(g)
    assert result['canAutoRun'], result['issues']
    assert result['loaders'][0]['settings']['attention_backend'] == '_native_math'
    assert result['loaders'][0]['settings']['vae_slicing'] is True
    assert result['resolvedFields']['load']['attention_backend'] == '_native_math'
    assert not result['patches']
    assert g == before


@pytest.mark.parametrize('field,value', [('attention_backend', 'flash'), ('vae_slicing', 'true'), ('vae_tiling', 1)])
def test_native_runtime_policy_invalid_values_cannot_be_ignored_by_auto(setup, field, value):
    plan, _, _, _ = setup
    g = graph()
    g['nodes']['load']['params'][field] = {'value': value}
    result = plan(g)
    assert not result['canAutoRun']
    assert result['patches'] == []


def test_native_runtime_policy_unknown_supplier_stays_unreviewed(setup):
    plan, _, _, _ = setup
    g = graph()
    g['nodes'] = {'policy': {'module': 'unreviewed.Source', 'action': 'Policy', 'params': {}}, **g['nodes']}
    g['nodes']['load']['params']['attention_backend'] = {'sourceId': 'policy', 'sourceKey': 'output'}
    g['paths'] = [list(g['nodes'])]
    result = plan(g)
    assert not result['canAutoRun']
    assert 'data-only preparation contract' in ' '.join(result['issues'])


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


def test_shared_cleanup_forecast_never_caps_accessible_free_memory_by_dedicated_vram(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate['requirements'] = {'systemRamBytes': 3 * gib, 'vramBytes': 3 * gib, 'diskFreeBytes': 0}
    hardware['systemMemory'].update(totalBytes=32 * gib, availableBytes=4 * gib)
    hardware['accelerator'].update(kind='cuda', memoryKind='shared', totalBytes=2 * gib, freeBytes=gib,
                                   accessibleTotalBytes=96 * gib)
    hardware['runtime'] = {'hardware': {'devices': [{'type': 'cuda', 'memory_kind': 'shared',
        'shared_memory_free': 4 * gib, 'torch_vram_free': 90 * gib}]}}
    result = plan(graph(), cache_snapshot={'reclaimable': {'systemRamBytes': 4 * gib, 'vramBytes': 0}})
    assert result['canAutoRun'], result['issues']
    assert result['requiresCachePreparation']
    assert result['available']['vramBytes'] == 4 * gib
    assert result['requirements']['systemRamBytes'] + result['requirements']['vramBytes'] == 6 * gib
    assert result['reusedOwnerIds'] == []  # Forecast is release/recheck, never resident-owner proof.


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


def native_lora_chain_graph(tmp_path, *, two_owners=False):
    import hashlib
    import numpy as np
    from safetensors.numpy import save_file

    path = tmp_path / "chain-adapter.safetensors"
    save_file({"lora_A.weight": np.zeros((2, 4), dtype=np.float32),
               "lora_B.weight": np.zeros((8, 2), dtype=np.float32)}, str(path))
    g = graph(two=two_owners)
    descriptors = {}
    for key, name in (("style1", "first_style"), ("style2", "second_style")):
        descriptors[key] = node("Lora", model={"source": "local", "value": str(path)},
                                weight_name=path.name, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                adapter_name=name, scale=1.0)
    descriptors["style2"]["params"]["previous_loras"] = {"sourceId": "style1", "sourceKey": "lora"}
    g["nodes"] = {**descriptors, **g["nodes"]}
    for key in ("load", "load2") if two_owners else ("load",):
        g["nodes"][key]["params"]["lora_list"] = {"sourceId": "style2", "sourceKey": "lora"}
    g["paths"] = [list(g["nodes"])]
    return g


def test_native_lora_chain_budgets_each_descriptor_once_in_order(setup, tmp_path):
    plan, _, hardware, _ = setup
    hardware["systemMemory"]["availableBytes"] = hardware["accelerator"]["freeBytes"] = 10000
    g = native_lora_chain_graph(tmp_path)
    before = deepcopy(g)
    result = plan(g)
    assert result["canAutoRun"], result["issues"]
    assert [item["nodeId"] for item in result["adapters"]] == ["style1", "style2"]
    assert [item["loaderIds"] for item in result["adapters"]] == [["load"], ["load"]]
    assert result["requirements"]["systemRamBytes"] == 300 + 544 * 2
    assert result["requirements"]["vramBytes"] == 200 + 544 * 2
    assert g == before


def test_native_lora_chain_shared_by_independent_owners_counts_storage_per_owner(setup, tmp_path):
    plan, _, hardware, _ = setup
    hardware["systemMemory"]["availableBytes"] = hardware["accelerator"]["freeBytes"] = 10000
    result = plan(native_lora_chain_graph(tmp_path, two_owners=True))
    assert result["canAutoRun"], result["issues"]
    assert all(item["loaderIds"] == ["load", "load2"] for item in result["adapters"])
    assert result["requirements"]["systemRamBytes"] == 300 * 2 + 544 * 4


@pytest.mark.parametrize("change", ["disconnected", "unknown_supplier", "wrong_output", "wrong_loader_port", "duplicate_name"])
def test_native_lora_chain_invalid_shapes_fail_before_auto(setup, tmp_path, change):
    plan, _, hardware, _ = setup
    hardware["systemMemory"]["availableBytes"] = hardware["accelerator"]["freeBytes"] = 10000
    g = native_lora_chain_graph(tmp_path)
    if change == "disconnected":
        del g["nodes"]["style2"]["params"]["previous_loras"]
    elif change == "unknown_supplier":
        g["nodes"]["style1"]["action"] = "UnknownDescriptor"
    elif change == "wrong_output":
        g["nodes"]["style2"]["params"]["previous_loras"]["sourceKey"] = "other"
    elif change == "wrong_loader_port":
        g["nodes"]["load"]["params"]["other_adapter"] = g["nodes"]["load"]["params"].pop("lora_list")
    else:
        g["nodes"]["style2"]["params"]["adapter_name"]["value"] = "first_style"
    result = plan(g)
    assert not result["canAutoRun"]
    assert result["patches"] == []
    assert any("adapter" in issue.lower() or "lora" in issue.lower() for issue in result["issues"])


def test_native_lora_chain_cycles_are_rejected_without_supplier_execution(setup, tmp_path):
    plan, _, _, _ = setup
    g = native_lora_chain_graph(tmp_path)
    g["nodes"]["style1"]["params"]["previous_loras"] = {"sourceId": "style2", "sourceKey": "lora"}
    with pytest.raises(ValueError, match="cycle|ordered after"):
        plan(g)


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


def test_split_decoder_dimension_connections_are_planned_without_executing_denoise(setup):
    plan, _, _, requests = setup
    g = graph()
    g['nodes']['generate']['action'] = 'Denoise'
    g['nodes']['generate']['params']['width']['value'] = '1024'
    g['nodes']['decode'] = node(
        'DecodeLatents', vae={'sourceId': 'load', 'sourceKey': 'vae_out'},
        latents={'sourceId': 'generate', 'sourceKey': 'latents'},
        width={'sourceId': 'generate', 'sourceKey': 'out_width'},
        height={'sourceId': 'generate', 'sourceKey': 'out_height'},
    )
    g['paths'] = [list(g['nodes'])]
    before = deepcopy(g)
    result = plan(g)
    assert result['canAutoRun'], result['issues']
    assert not result['requiresPreparation']
    # The executor rechecks these integer outputs against its actual connected
    # arguments before allocating decoder resources; no cached latent is read.
    assert result['resolvedFields']['decode'] == {'width': 1024, 'height': 1024}
    assert requests[-1]['form']['width'] == 1024
    assert g == before


@pytest.mark.parametrize('change', ['different_latents', 'unknown_output', 'non_integer'])
def test_split_decoder_dimension_projection_does_not_authorize_unknown_geometry(setup, change):
    plan, _, _, _ = setup
    g = graph()
    g['nodes']['generate']['action'] = 'Denoise'
    g['nodes']['decode'] = node(
        'DecodeLatents', vae={'sourceId': 'load', 'sourceKey': 'vae_out'},
        latents={'sourceId': 'generate', 'sourceKey': 'latents'},
        width={'sourceId': 'generate', 'sourceKey': 'out_width'},
    )
    if change == 'different_latents':
        g['nodes']['decode']['params']['latents'] = {'sourceId': 'load', 'sourceKey': 'latents'}
    elif change == 'unknown_output':
        g['nodes']['decode']['params']['width']['sourceKey'] = 'estimated_width'
    else:
        g['nodes']['generate']['params']['width']['value'] = 1.5
    g['paths'] = [list(g['nodes'])]
    assert not plan(g)['canAutoRun']


def test_split_decoder_stops_before_allocation_if_actual_geometry_differs():
    from modiff.server import WebServer

    app = object.__new__(WebServer)
    app.modules = {'modules.ModularDiffusers': {'DecodeLatents': {}}}
    app.node_cache = {'denoise': SimpleNamespace(output={'out_width': 2048})}
    app._workflow_auto_resolved_fields = {'decode': {'width': 1024}}
    with pytest.raises(RuntimeError, match='decode.width changed after resource planning'):
        app.execute_node('decode', node('DecodeLatents', width={
            'sourceId': 'denoise', 'sourceKey': 'out_width',
        }), 'test')
    assert 'decode' not in app.node_cache


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
    server._build_workflow_auto_plan = lambda g, **kwargs: {'canAutoRun': True, 'requiresPreparation': False, 'preparationNodeIds': [], 'seen': RUNTIME_VALUES.get()[('source', 'width')]}
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


@pytest.mark.parametrize("shared, ram, vram, release", [
    (False, 1000, 1000, False), (False, 450, 1000, True),
    (False, 1000, 300, True), (True, 1000, 1000, False),
    (True, 800, 1000, True),
])
def test_owner_retention_uses_the_combined_live_memory_envelope(setup, shared, ram, vram, release):
    plan, _, hardware, _ = setup
    hardware['systemMemory']['availableBytes'] = ram
    hardware['accelerator'].update(freeBytes=vram, memoryKind='shared' if shared else 'dedicated')
    result = plan(graph(two=True))
    assert result['canAutoRun']
    assert (result['schedule'] is not None) is release
    assert result['retainedRequirements']['systemRamBytes'] == 600


def test_owner_retention_normalizes_numeric_hardware_values(setup):
    plan, _, hardware, _ = setup
    hardware['systemMemory']['availableBytes'] = '1000'
    hardware['accelerator']['freeBytes'] = '1000'
    result = plan(graph(two=True))
    assert result['canAutoRun']
    assert result['schedule'] is None


def test_unqualified_recipe_explains_installed_files_are_not_auto_proof(setup):
    plan, candidate, _, _ = setup
    candidate['canAutoRun'] = False
    g = graph()
    before = deepcopy(g)
    result = plan(g)
    assert not result['canAutoRun']
    message = ' '.join(result['issues'])
    assert 'Installed model files do not establish Auto resource qualification' in message
    assert 'test/model' in message
    assert 'bfloat16' in message
    assert 'Custom memory' in message
    assert g == before


def warm_cache(g, *, ram=6 * 1024 ** 3, vram=0):
    return {"owners": {"load": {"cacheKey": planner.workflow_owner_cache_key(g, "load"),
                                 "systemRamBytes": ram, "vramBytes": vram}},
            "reclaimable": {"systemRamBytes": ram, "vramBytes": vram}}


def test_warm_workflow_reuses_resident_weights_without_counting_full_ram_again(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate['requirements'] = {'systemRamBytes': 8 * gib, 'vramBytes': 0}
    hardware['systemMemory'].update(totalBytes=32 * gib, availableBytes=12 * gib)
    hardware['accelerator']['freeBytes'] = 16 * gib
    g = graph()
    cold = plan(g)
    assert cold['canAutoRun']
    hardware['systemMemory']['availableBytes'] = int(4.5 * gib)
    # The loader/weights are unchanged while the prompt consumer changes.
    g['nodes']['generate']['params']['prompt'] = {'value': 'a new image'}
    warm = plan(g, cache_snapshot=warm_cache(g))
    assert warm['canAutoRun'], warm['issues']
    assert warm['requirements']['systemRamBytes'] == 4 * gib
    assert warm['retainedRequirements']['systemRamBytes'] == 8 * gib
    assert warm['reusedOwnerIds'] == ['load']
    assert not warm['requiresCachePreparation']


def test_reported_warm_ram_pressure_plans_cleanup_before_rechecking_capacity(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate['requirements'] = {'systemRamBytes': 8 * gib, 'vramBytes': 0}
    hardware['systemMemory'].update(totalBytes=32 * gib, availableBytes=4279840768)
    hardware['accelerator']['freeBytes'] = 16 * gib
    g = graph()
    result = plan(g, cache_snapshot=warm_cache(g))
    assert result['canAutoRun'], result['issues']
    assert result['requiresCachePreparation']
    assert result['reusedOwnerIds'] == []
    assert result['requirements']['systemRamBytes'] == 8 * gib
    assert result['available']['systemRamBytes'] == 4279840768
    assert 'recheck' in result['message'].lower()


def test_resident_credit_does_not_authorize_external_memory_pressure(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate['requirements'] = {'systemRamBytes': 8 * gib, 'vramBytes': 0}
    hardware['systemMemory'].update(totalBytes=32 * gib, availableBytes=gib)
    hardware['accelerator']['freeBytes'] = 16 * gib
    g = graph()
    result = plan(g, cache_snapshot=warm_cache(g, ram=2 * gib))
    assert not result['canAutoRun']
    assert not result['requiresCachePreparation']
    assert 'System RAM' in ' '.join(result['issues'])
    assert 'GiB' in ' '.join(result['issues'])


@pytest.mark.parametrize('kind', ['cuda', 'mps', 'xpu'])
def test_shared_memory_reuse_keeps_combined_host_and_accelerator_headroom(setup, kind):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate['requirements'] = {'systemRamBytes': 8 * gib, 'vramBytes': 6 * gib}
    hardware['systemMemory'].update(totalBytes=32 * gib, availableBytes=7 * gib)
    hardware['accelerator'].update(kind=kind, totalBytes=16 * gib, freeBytes=7 * gib, memoryKind='shared',
                                  sharedMemoryAvailableBytes=7 * gib)
    g = graph()
    g['nodes']['load']['params']['device']['value'] = 'mps' if kind == 'mps' else f'{kind}:0'
    result = plan(g, cache_snapshot=warm_cache(g, ram=6 * gib, vram=6 * gib))
    assert result['canAutoRun'], result['issues']
    assert result['sharedMemory'] and result['reusedOwnerIds'] == ['load']
    assert sum(result['requirements'][key] for key in ('systemRamBytes', 'vramBytes')) <= 7 * gib
    hardware['systemMemory']['availableBytes'] = 5 * gib
    hardware['accelerator'].update(freeBytes=5 * gib, sharedMemoryAvailableBytes=5 * gib)
    pressured = plan(g, cache_snapshot=warm_cache(g, ram=6 * gib, vram=6 * gib))
    assert pressured['requiresCachePreparation'] and pressured['reusedOwnerIds'] == []


def test_live_vram_storage_is_credited_once_and_external_gpu_pressure_still_blocks(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate['requirements'] = {'systemRamBytes': 0, 'vramBytes': 8 * gib}
    hardware['systemMemory'].update(totalBytes=32 * gib, availableBytes=12 * gib)
    hardware['accelerator'].update(totalBytes=16 * gib, freeBytes=gib)
    result = plan(graph(), cache_snapshot=warm_cache(graph(), ram=0, vram=6 * gib))
    assert not result['canAutoRun'] and not result['requiresCachePreparation']
    assert result['available']['vramBytes'] == gib
    assert result['requirements']['vramBytes'] == 2 * gib
    assert 'Accelerator memory' in ' '.join(result['issues'])


def test_gpu_only_warm_owner_cannot_invent_host_loading_credit(setup):
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate['requirements'] = {'systemRamBytes': 8 * gib, 'vramBytes': 8 * gib}
    hardware['systemMemory'].update(totalBytes=32 * gib, availableBytes=6 * gib)
    hardware['accelerator'].update(totalBytes=16 * gib, freeBytes=4 * gib)
    g = graph()
    result = plan(g, cache_snapshot=warm_cache(g, ram=0, vram=6 * gib))
    assert not result['canAutoRun'] and not result['requiresCachePreparation']
    assert result['reusedOwnerIds'] == ['load']
    assert result['requirements']['vramBytes'] == 2 * gib
    # The aggregate recipe declares no separate transient host-loading demand.
    # Known CUDA weights cannot reduce unmeasured host inference requirements.
    assert result['requirements']['systemRamBytes'] == 8 * gib
    assert result['available']['systemRamBytes'] == 6 * gib
    assert 'System RAM' in ' '.join(result['issues'])


@pytest.mark.parametrize('field,value', [('dtype', 'float16'), ('offload_mode', 'model_cpu'),
                                       ('revision', 'a' * 40)])
def test_owner_cache_identity_changes_with_loader_settings_but_not_consumers(field, value):
    g = graph()
    original = planner.workflow_owner_cache_key(g, 'load')
    g['nodes']['generate']['params']['num_inference_steps']['value'] = 10
    assert planner.workflow_owner_cache_key(g, 'load') == original
    g['nodes']['load']['params'][field] = {'value': value}
    assert planner.workflow_owner_cache_key(g, 'load') != original


def test_owner_cache_identity_includes_connected_adapter_source():
    g = graph()
    g['nodes']['adapter'] = {'module': 'modules.ModularDiffusers', 'action': 'Lora',
                             'params': {'repo_id': {'value': 'test/lora'}, 'scale': {'value': 1}}}
    g['nodes']['load']['params']['lora'] = {'sourceId': 'adapter', 'sourceKey': 'lora'}
    g['paths'] = [['adapter', 'load', 'generate', 'preview']]
    original = planner.workflow_owner_cache_key(g, 'load')
    g['nodes']['adapter']['params']['scale']['value'] = .5
    assert planner.workflow_owner_cache_key(g, 'load') != original


def test_owner_cache_identity_includes_actual_connected_resource_values():
    g = graph()
    g['nodes']['source'] = {'module': 'modules.Primitive', 'action': 'String',
                             'params': {'value': {'value': 'test/model'}}}
    g['nodes']['load']['params']['repo_id'] = {'sourceId': 'source', 'sourceKey': 'output'}
    first = planner.workflow_owner_cache_key(g, 'load', {'load': {'repo_id': 'test/model'}})
    assert planner.workflow_owner_cache_key(g, 'load', {'load': {'repo_id': 'test/other'}}) != first


def test_cache_snapshot_counts_unique_storage_and_does_not_retain_model(tmp_path, monkeypatch):
    import sys
    import weakref
    import torch
    from modiff.server import WebServer, memory_manager

    class Loader:
        pass

    class Model(torch.nn.Module):
        def parameters(self, *args, **kwargs):
            raise AssertionError('inspection must not invoke model overrides')

    model = Model()
    model.weight = torch.nn.Parameter(torch.zeros(16))
    model.register_buffer('same_storage', model.weight.detach())
    cached = Loader()
    cached._cache_valid, cached._cache_invalidated = True, False
    cached._mm_models, cached.output = ['weights'], {'model': model}
    cached.loader = model
    app = object.__new__(WebServer)
    app.node_cache = {'load': cached}
    monkeypatch.setattr(memory_manager, 'cache', {'weights': {'model': model}})
    monkeypatch.setitem(sys.modules, 'modules.ModularDiffusers', SimpleNamespace(
        components=SimpleNamespace(collections={'load': ['component']}, components={'component': model})))
    owner = {'nodeId': 'load', 'cacheKey': 'reviewed-owner'}
    app._record_workflow_auto_owner(owner)
    snapshot = app._workflow_auto_cache_snapshot()
    assert snapshot['reclaimable'] == {'systemRamBytes': 64, 'vramBytes': 0}
    assert snapshot['owners']['load'] == {'cacheKey': 'reviewed-owner', 'systemRamBytes': 64, 'vramBytes': 0}
    cached.output = {'model': model}
    assert not app._workflow_auto_cache_snapshot()['owners']
    app._record_workflow_auto_owner(owner)
    cached._cache_invalidated = True
    assert not app._workflow_auto_cache_snapshot()['owners']
    assert app._workflow_auto_owner_records == {}
    reference = weakref.ref(cached)
    app.node_cache.clear()
    del cached
    assert reference() is None


def test_workflow_inspection_and_dispatch_both_receive_owner_storage(tmp_path, monkeypatch):
    from modiff.server import WebServer
    import modiff.server as server_module
    app = object.__new__(WebServer)
    app.current_task = None
    app.data_dir = str(tmp_path)
    app._auto_resource_runtime_block = lambda: None
    app._runtime_fingerprint = lambda: {'fingerprint': 'actual-free'}
    app._auto_planning_runtime_fingerprint = lambda: pytest.fail('owner-aware workflow must not add all CUDA reservations')
    cache = {'owners': {'load': {'systemRamBytes': 6 * 1024 ** 3}}, 'reclaimable': {}}
    app._workflow_auto_cache_snapshot = lambda: cache
    monkeypatch.setattr(server_module, 'get_local_models', lambda: [])
    seen = []
    monkeypatch.setattr(server_module, 'build_workflow_auto_plan', lambda g, **kw: seen.append(kw) or {'canAutoRun': True})
    app._build_workflow_auto_plan(graph())
    app.current_task = {'task_id': 'dispatch'}
    app._build_workflow_auto_plan(graph(), dispatch=True)
    assert all(item['cache_snapshot'] is cache for item in seen)
    assert all(item['runtime_fingerprint']['fingerprint'] == 'actual-free' for item in seen)
    app._auto_planning_runtime_fingerprint = lambda: {'fingerprint': 'before-active-run'}
    app._workflow_auto_cache_snapshot = lambda: pytest.fail('active HTTP inspection must not walk loading models')
    app._runtime_fingerprint = lambda: pytest.fail('active HTTP inspection must not enter accelerator probes')
    app._build_workflow_auto_plan(graph())
    assert seen[-1]['cache_snapshot'] == {}


def test_cache_preparation_keeps_prepared_data_and_rechecks_its_actual_values():
    from modiff.server import WebServer
    from modiff.workflow_auto_values import RUNTIME_VALUES
    app = object.__new__(WebServer)
    prepared = SimpleNamespace(output={'width': 1024})
    app.node_cache, app.current_task = {'source': prepared, 'old': object()}, None
    messages = []
    app.queue_message = messages.append
    def release():
        app.node_cache.clear()
        return {'released': {'nodes': 2}, 'errors': []}
    app._release_runtime_caches_for_retry = release
    def replan(graph, **kwargs):
        assert kwargs == {'dispatch': True}
        assert app.node_cache == {'source': prepared}
        assert RUNTIME_VALUES.get() == {('source', 'width'): 1024}
        return {'canAutoRun': True, 'requiresCachePreparation': False, 'available': {}}
    app._build_workflow_auto_plan = replan
    result, cleanup = app._prepare_workflow_auto_cache({'sid': 'unit'},
        {'canAutoRun': True, 'requiresCachePreparation': True}, {'source'})
    assert result['canAutoRun'] and cleanup['released']['nodes'] == 2
    assert app.node_cache['source'] is prepared and not RUNTIME_VALUES.get()
    assert [item['performed'] for item in messages] == [False, True]


@pytest.mark.parametrize('stale', ['source_invalidated', 'source_replaced', 'source_output',
                                   'loader_inputs', 'loader_implementation', 'model_evicted'])
def test_owner_credit_requires_ordinary_loader_and_upstream_cache_hits(monkeypatch, stale):
    import torch
    from modiff.server import WebServer, memory_manager
    from modiff.node_cache_identity import implementation_identity, input_snapshot
    class Cached:
        CALLBACK = 'execute'
        def execute(self):
            pass
    source, loader = Cached(), Cached()
    for node in (source, loader):
        node._cache_valid, node._cache_invalidated = True, False
        node.params, node.output = {'revision': 'original'}, {'value': 'original'}
        node._cache_input_snapshot = input_snapshot(node.params)
        node._cache_implementation = implementation_identity(node)
        node._mm_models, node._cache_input_sources = [], ()
    loader._cache_input_sources = ('source',)
    loader._mm_models = ['weights']
    model = torch.nn.Linear(4, 4, bias=False)
    monkeypatch.setattr(memory_manager, 'cache', {'weights': {'model': model}})
    app = object.__new__(WebServer)
    app.node_cache = {'load': loader, 'source': source}
    app._record_workflow_auto_owner({'nodeId': 'load', 'cacheKey': 'owner'})
    assert app._workflow_auto_cache_snapshot()['owners']['load']['systemRamBytes'] == 64
    if stale == 'source_invalidated':
        source._cache_invalidated = True
    elif stale == 'source_replaced':
        app.node_cache['source'] = Cached()
    elif stale == 'source_output':
        source.output = {'value': 'changed'}
    elif stale == 'loader_inputs':
        loader.params['revision'] = 'changed'
    elif stale == 'loader_implementation':
        loader._cache_implementation = None, None
    else:
        memory_manager.cache.clear()
    assert not app._workflow_auto_cache_snapshot()['owners']


@pytest.mark.parametrize('capacity_semantics', [False, True])
def test_five_warm_dispatches_reuse_cpu_weights_and_generate_each_changed_prompt(setup, tmp_path, monkeypatch, capacity_semantics):
    import torch
    from modiff.server import WebServer, memory_manager
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate['requirements'] = {'systemRamBytes': 8 * gib, 'vramBytes': 0}
    if capacity_semantics:
        candidate['requirements'] = {'memorySemantics': 'machine_capacity', 'minimum': candidate['requirements']}
    hardware['systemMemory'].update(totalBytes=32 * gib, availableBytes=12 * gib)
    hardware['accelerator']['freeBytes'] = 16 * gib
    model = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    monkeypatch.setattr(memory_manager, 'cache', {'weights': {'model': model}})
    class Loader:
        pass
    cached = Loader()
    cached._cache_valid, cached._cache_invalidated = True, False
    cached._mm_models, cached.output = ['weights'], {'model': model}
    app = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    app.current_task = {'task_id': 'five-warm-runs', 'progress': 0}
    app._build_workflow_auto_plan = lambda graph, **kwargs: plan(graph, cache_snapshot=app._workflow_auto_cache_snapshot())
    app._prepare_auto_runtime_for_graph = lambda _: None
    app._runtime_fingerprint = lambda: {'fingerprint': 'unit'}
    app._runtime_measurement = lambda **_: {'elapsedSeconds': 0}
    app._record_auto_resource_success = lambda *a, **k: None
    app._record_optimization_observations = lambda *a, **k: []
    messages, generations, outputs, weight_loads = [], [], [], []
    app.queue_message = messages.append
    def execute(node_id, node, sid):
        if node_id == 'load':
            if node_id not in app.node_cache:
                weight_loads.append(node_id)
                app.node_cache[node_id] = cached
                # Real CPU computations use controlled memory samples. The
                # capacity tier is never an additional free-RAM requirement.
                hardware['systemMemory']['availableBytes'] = 5 * gib if capacity_semantics else 8 * gib - 32
        elif node_id == 'generate':
            generations.append((node['params']['prompt']['value'], node['params']['seed']['value']))
            with torch.no_grad():
                outputs.append(model(torch.full((1, 4), float(node['params']['seed']['value'] + 1))).tolist())
    app.execute_node = execute
    for index in range(5):
        g = graph()
        g['sid'] = 'unit'
        g['nodes']['generate']['params'].update(prompt={'value': f'image {index}'}, seed={'value': index})
        g['runtimeHints'] = {'resourceMode': 'auto', 'workflowAutoPlan': {'schemaVersion': 1, 'graphHash': planner.workflow_graph_hash(g)}}
        app._execute_graph(g)
    assert weight_loads == ['load']
    assert generations == [(f'image {index}', index) for index in range(5)]
    assert outputs == [[[4.0 * (index + 1)] * 4] for index in range(5)]
    completed = [item for item in messages if item['type'] == 'graph_completed']
    assert len(completed) == 5
    assert all(item['runtimePreparation']['workflowAuto']['reusedOwnerIds'] == ['load'] for item in completed[1:])


@pytest.mark.parametrize('memory_after_release', [12, 1])
def test_dispatch_releases_cache_before_capacity_gate_and_requires_actual_ram(setup, tmp_path, memory_after_release):
    from modiff.server import WebServer
    plan, candidate, hardware, _ = setup
    gib = 1024 ** 3
    candidate['requirements'] = {'systemRamBytes': 8 * gib, 'vramBytes': 0}
    hardware['systemMemory'].update(totalBytes=32 * gib, availableBytes=4279840768)
    hardware['accelerator']['freeBytes'] = 16 * gib
    g = graph()
    g['sid'] = 'unit'
    g['runtimeHints'] = {'resourceMode': 'auto', 'workflowAutoPlan': {'schemaVersion': 1, 'graphHash': planner.workflow_graph_hash(g)}}
    app = WebServer(modules={}, work_dir=str(tmp_path), data_dir=str(tmp_path))
    app.current_task = {'task_id': 'warm-test', 'progress': 0}
    calls, messages = [], []
    cache = warm_cache(g)
    app._build_workflow_auto_plan = lambda graph, **kwargs: plan(graph, cache_snapshot=cache)
    def release():
        calls.append('release')
        cache.clear()
        hardware['systemMemory']['availableBytes'] = memory_after_release * gib
        return {'errors': [], 'released': {'models': 1}}
    app._release_runtime_caches_for_retry = release
    app.queue_message = messages.append
    app._prepare_auto_runtime_for_graph = lambda _: None
    app._runtime_fingerprint = lambda: {'fingerprint': 'unit'}
    app._runtime_measurement = lambda **_: {'elapsedSeconds': 0}
    app._record_auto_resource_success = lambda *a, **k: None
    app._record_optimization_observations = lambda *a, **k: []
    app.execute_node = lambda id, *args: calls.append(id)
    if memory_after_release == 1:
        with pytest.raises(RuntimeError, match='System RAM.*8.00 GiB.*1.00 GiB'):
            app._execute_graph(g)
        assert calls == ['release']
    else:
        app._execute_graph(g)
        assert calls == ['release', 'load', 'generate', 'preview']
        completed = next(item for item in messages if item['type'] == 'graph_completed')
        assert completed['runtimePreparation']['workflowAuto']['cachePreparation']['released']['models'] == 1
