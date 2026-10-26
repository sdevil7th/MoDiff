from copy import deepcopy

import pytest

from modiff.workflow_auto_resource import build_workflow_auto_plan


@pytest.mark.parametrize('pipeline', ['Flux2KleinModularPipeline', 'FluxKontextModularPipeline'])
@pytest.mark.parametrize('conflicting_explicit_modes', [False, True])
@pytest.mark.parametrize('registered_profile', [False, True])
def test_inferred_reviewed_edit_aliases_use_conservative_resource_mode(
    monkeypatch, tmp_path, pipeline, conflicting_explicit_modes, registered_profile,
):
    from types import SimpleNamespace
    from modiff import workflow_auto_resource as planner

    graph = modular_graph(pipeline, 'edit_image')
    if conflicting_explicit_modes:
        graph['nodes']['loader']['params']['mode'] = {'value': 'edit_image'}
        graph['nodes']['denoise']['params']['mode'] = {'value': 'multi_image_reference_edit'}
    before = deepcopy(graph)
    profile = SimpleNamespace(
        id='test:reviewed-edit', model_type=pipeline,
        modes=('edit_image', 'multi_image_reference_edit'),
        loader_module='modules.ModularDiffusers', loader_action='ModelsLoader',
        execution_path='modular-diffusers',
    )
    if registered_profile:
        if pipeline == 'Flux2KleinModularPipeline':
            from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES

            registered = DIFFUSERS_EXECUTION_PROFILES['flux2-klein:modular']
            graph['nodes']['loader']['params']['execution_profile_id'] = {'value': 'flux2-klein:modular'}
            graph['nodes']['loader']['params']['repo_id'] = {'value': registered.default_repo}
        profiles, reason = planner.resolve_execution_profiles_for_loader(
            'modules.ModularDiffusers', 'ModelsLoader',
            {field: param['value'] for field, param in graph['nodes']['loader']['params'].items()},
        )
        assert reason is None and len(profiles) == 1
        profile = profiles[0]
        before = deepcopy(graph)
    monkeypatch.setattr(
        planner, 'resolve_execution_profiles_for_loader',
        lambda module, action, values: ((profile,), None) if action == 'ModelsLoader' else ((), None),
    )
    requests = []

    def recipe(payload, **kwargs):
        form = payload['form']
        requests.append(deepcopy(form))
        return {'candidates': [{
            'id': 'reviewed-edit', 'executionProfileId': profile.id,
            'modelRepo': form['modelRepo'], 'modelType': pipeline,
            'mode': form['mode'], 'loaderModule': profile.loader_module,
            'loaderAction': profile.loader_action, 'executionPath': profile.execution_path,
            'dtype': 'float32', 'quantizationMode': 'none', 'offloadMode': 'none',
            'autoOffload': False, 'canAutoRun': True, 'proof': {'status': 'declared_safe'},
            'requirements': {'systemRamBytes': 1, 'vramBytes': 0, 'diskFreeBytes': 0},
        }]}

    result = planner.build_workflow_auto_plan(
        graph, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path),
        plan_recipe=recipe,
        hardware={'systemMemory': {'availableBytes': 1000},
                  'accelerator': {'freeBytes': 1000}, 'offloadDisk': {'freeBytes': 1000}},
    )
    assert graph == before
    if conflicting_explicit_modes:
        assert not result['canAutoRun'] and any('ambiguous' in issue for issue in result['issues'])
        assert requests == []
    else:
        assert result['canAutoRun'], result['issues']
        expected = 'multi_image_reference_edit' if 'multi_image_reference_edit' in profile.modes else 'edit_image'
        assert requests[0]['mode'] == expected


def modular_graph(pipeline, mode):
    from modiff.modular_action_bindings import MODULAR_ACTION_BINDINGS
    from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH

    route = PINNED_MODULAR_WORKFLOW_TRUTH[pipeline].mode(mode)
    nodes = {
        "loader": {
            "module": "modules.ModularDiffusers",
            "action": "ModelsLoader",
            "params": {
                "model_type": {"value": pipeline},
                "repo_id": {"value": "test/model"},
                "device": {"value": "cpu"},
                "dtype": {"value": "float32"},
            },
        }
    }
    for action in route.action_sequence:
        module, name = MODULAR_ACTION_BINDINGS[action][1].rsplit(".", 1)
        nodes[action] = {
            "module": module,
            "action": name,
            "params": {
                "pipeline_components": {"sourceId": "loader", "sourceKey": "pipeline_components"},
            },
        }
    for edge in route.state_edges:
        nodes[edge.consumer_action]["params"][edge.consumer_input] = {
            "sourceId": edge.producer_action,
            "sourceKey": edge.producer_output,
        }
    # Only the task's actual external input set is supplied.
    for name in route.required_upstream_inputs:
        nodes[route.action_sequence[0]]["params"][name] = {"value": "test-input"}
    return {"nodes": nodes, "paths": [list(nodes)]}


def test_auto_resolves_sdxl_img2img_from_the_executable_graph(monkeypatch, tmp_path):
    from modiff import workflow_auto_resource as planner
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES

    profile = DIFFUSERS_EXECUTION_PROFILES["sdxl-base:modular"]
    monkeypatch.setattr(
        planner,
        "resolve_execution_profiles_for_loader",
        lambda module, action, values: ((profile,), None) if action == "ModelsLoader" else ((), None),
    )
    requests = []

    def recipe(payload, **kwargs):
        requests.append(payload)
        return {"candidates": []}

    graph = modular_graph(profile.model_type, "image_to_image")
    before = deepcopy(graph)
    result = build_workflow_auto_plan(
        graph,
        runtime_fingerprint={},
        local_models=[],
        data_dir=str(tmp_path),
        plan_recipe=recipe,
        hardware={"systemMemory": {}, "accelerator": {}, "offloadDisk": {}},
    )
    # The actual registered profile names this upstream image-to-image task
    # edit_image; task discovery must retain that exact recipe identity.
    assert requests and requests[0]["form"]["mode"] == "edit_image"
    assert requests[0]["form"]["executionProfileId"] == profile.id
    assert not result["canAutoRun"], "Resolving a task does not grant an unqualified resource recipe."
    assert graph == before


@pytest.mark.parametrize(
    "pipeline,task",
    [
        ("StableDiffusionXLModularPipeline", "text_to_image"),
        ("StableDiffusionXLModularPipeline", "image_to_image"),
        ("StableDiffusionXLModularPipeline", "inpaint"),
        ("FluxModularPipeline", "text_to_image"),
        ("FluxModularPipeline", "image_to_image"),
        ("QwenImageModularPipeline", "text_to_image"),
    ],
)
def test_reviewed_tasks_are_identified_without_authoring_hints(pipeline, task):
    from modiff.workflow_task_identity import modular_graph_tasks

    graph = modular_graph(pipeline, task)
    assert modular_graph_tasks(graph["nodes"], "loader", list(graph["nodes"])[1:], pipeline) == {task}


def test_missing_or_wrong_state_wire_and_duplicate_consumers_do_not_guess():
    from modiff.workflow_task_identity import modular_graph_tasks

    pipeline = "StableDiffusionXLModularPipeline"
    graph = modular_graph(pipeline, "image_to_image")["nodes"]

    def tasks(nodes):
        return modular_graph_tasks(nodes, "loader", list(nodes)[1:], pipeline)

    assert tasks(graph) == {"image_to_image"}
    broken = deepcopy(graph)
    edge = next(p for p in broken["denoise"]["params"].values() if p.get("sourceId") == "vae_encoder")
    edge["sourceKey"] = "unrelated"
    assert tasks(broken) == set()
    duplicate = deepcopy(graph)
    duplicate["other_denoise"] = deepcopy(graph["denoise"])
    assert tasks(duplicate) == set()
    assert modular_graph_tasks(graph, "loader", ["denoise"], pipeline) == set()
    assert modular_graph_tasks(graph, "loader", list(graph)[1:], "UnknownPipeline") == set()


def test_guidance_helpers_belong_to_the_connected_owner_without_crossing_media():
    from modiff.workflow_task_identity import resource_consumers

    nodes = modular_graph("StableDiffusionXLModularPipeline", "image_to_image")["nodes"]
    nodes["layers"] = {"module": "modules.ModularDiffusers", "action": "Layers", "params": {}}
    nodes["guide"] = {"module": "modules.ModularDiffusers", "action": "Guider", "params": {
        "layers_config": {"sourceId": "layers", "sourceKey": "layers_config"},
    }}
    nodes["denoise"]["params"]["guider"] = {"sourceId": "guide", "sourceKey": "guider_out"}
    nodes["other"] = {"module": "modules.ModularDiffusers", "action": "ModelsLoader", "params": {}}
    nodes["image"] = {"module": "modules.ModularDiffusers", "action": "DecodeLatents", "params": {
        "vae": {"sourceId": "other", "sourceKey": "vae_out"},
    }}
    nodes["denoise"]["params"]["image"] = {"sourceId": "image", "sourceKey": "image"}
    found = resource_consumers(nodes, "loader", {"loader", "other"})
    assert {"guide", "layers"}.issubset(found)
    assert not {"image", "other"}.intersection(found)
    nodes["guide"]["module"] = "custom.Arbitrary"
    assert "guide" not in resource_consumers(nodes, "loader", {"loader", "other"})


def test_actual_operation_starter_edges_resolve_the_selected_image_task():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    from modiff.workflow_task_identity import modular_graph_tasks

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    for pipeline, task in [
        ("StableDiffusionXLModularPipeline", "text_to_image"),
        ("StableDiffusionXLModularPipeline", "image_to_image"),
        ("FluxModularPipeline", "text_to_image"),
        ("FluxModularPipeline", "image_to_image"),
        ("QwenImageModularPipeline", "text_to_image"),
    ]:
        starter = resolve_operation_starter(MODULE_MAP, contracts, {"pipelineClass": pipeline, "task": task})
        nodes = {
            n["operation"]["operationId"]: {
                "module": n["module"],
                "action": n["action"],
                "params": {
                    k: {"value": p.get("value", p.get("default"))}
                    for k, p in n["params"].items()
                    if p.get("display") != "output"
                },
            }
            for n in starter["nodes"]
        }
        for e in starter["edges"]:
            nodes[e["target"]]["params"][e["targetHandle"]] = {"sourceId": e["source"], "sourceKey": e["sourceHandle"]}
        loader = starter["nodes"][0]["operation"]["operationId"]
        assert modular_graph_tasks(nodes, loader, list(nodes)[1:], pipeline) == {task}, (pipeline, task)


def test_output_history_uses_recognized_graph_task_without_rewriting_form():
    from modiff.execution_input_provenance import (
        capture_generation_inputs,
        build_resolved_execution_inputs,
        apply_resolved_execution_inputs,
    )

    graph = modular_graph("StableDiffusionXLModularPipeline", "image_to_image")
    records = {
        key: capture_generation_inputs(
            key, node, {field: p.get("value") for field, p in node["params"].items() if "value" in p}
        )
        for key, node in graph["nodes"].items()
    }
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=0, node_id="decoder")
    output = {"taskId": "run", "nodeId": "decoder", "mode": "text_to_image", "formSnapshot": {"mode": "text_to_image"}}
    result = apply_resolved_execution_inputs(output, receipt)
    assert result["mode"] == "image_to_image"
    assert result["formSnapshot"]["mode"] == "text_to_image"
    assert receipt["graphTasks"] == [
        {"loaderId": "loader", "pipelineClass": "StableDiffusionXLModularPipeline", "task": "image_to_image"}
    ]


@pytest.mark.parametrize('module,loader,consumer,pipeline,task', [
    ('DiffusersImage', 'LoadPipeline', 'Inpaint', 'ZImageInpaintPipeline', 'inpaint'),
    ('DiffusersImage', 'LoadPipeline', 'Inpaint', 'ZImageInpaintPipeline', 'outpaint'),
    ('DiffusersImage', 'LoadPipeline', 'PredictMap', 'MarigoldDepthPipeline', 'depth_estimation'),
    ('DiffusersImage', 'LoadPipeline', 'UnconditionalGenerate', 'DDPMPipeline', 'unconditional_image'),
    ('HuggingFaceTransformers', 'LoadDepthEstimationModel', 'PredictDepth', 'AutoModelForDepthEstimation', 'depth_estimation'),
    ('HuggingFaceTransformers', 'LoadImageTextToTextModel', 'GenerateImageVideoText', 'AutoModelForImageTextToText', 'image_to_text'),
    ('ImageOperations', 'ProcessImage', None, 'BuiltinImageOperationV1', 'image_crop'),
])
def test_ordinary_image_receipts_use_validated_consumer_task(module, loader, consumer, pipeline, task):
    from modiff.execution_input_provenance import capture_generation_inputs, build_resolved_execution_inputs
    from modiff.workflow_task_identity import graph_task_receipts

    nodes = {'owner': {'module': 'modules.' + module, 'action': loader, 'params': {}}}
    records = {'owner': capture_generation_inputs('owner', nodes['owner'], {'pipeline_class': pipeline})}
    output_id = 'owner'
    if consumer:
        output_id = 'consumer'
        nodes[output_id] = {'module': 'modules.' + module, 'action': consumer, 'params': {
            'pipeline' if loader != 'LoadImageTextToTextModel' else 'model': {
                'sourceId': 'owner', 'sourceKey': 'pipeline' if loader != 'LoadImageTextToTextModel' else 'model'}}}
        records[output_id] = capture_generation_inputs(output_id, nodes[output_id], {'executed_task': task})
    else:
        records['owner'] = capture_generation_inputs('owner', nodes['owner'], {
            'pipeline_class': pipeline, 'executed_task': task})
    expected = [{'loaderId': 'owner', 'pipelineClass': pipeline, 'task': task}]
    assert graph_task_receipts(nodes, records) == expected
    receipt = build_resolved_execution_inputs({'nodes': nodes}, records, task_id='run', attempt_index=0, node_id=output_id)
    assert receipt['graphTasks'] == expected
    # An authoring label cannot rescue absent or conflicting consumed evidence.
    nodes[output_id]['params']['mode'] = {'value': task}
    records[output_id]['fields'].pop('executed_task')
    assert graph_task_receipts(nodes, records)[0]['task'] is None
    assert graph_task_receipts(nodes, {}) == []
    if consumer:
        records[output_id] = capture_generation_inputs(output_id, nodes[output_id], {'executed_task': 'video_to_video'})
        assert graph_task_receipts(nodes, records)[0]['task'] is None



@pytest.mark.parametrize('profile_identity', [{}, {'id': 'generic-single-mode-profile'}],
                         ids=['missing-identity', 'unrelated-identity'])
def test_conflicting_explicit_modes_remain_unresolved_even_for_a_single_mode_profile(monkeypatch, tmp_path, profile_identity):
    from types import SimpleNamespace
    from modiff import workflow_auto_resource as planner

    profile = SimpleNamespace(
        model_type="StableDiffusionXLModularPipeline",
        modes=("text_to_image",),
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
        **profile_identity,
    )
    monkeypatch.setattr(
        planner,
        "resolve_execution_profiles_for_loader",
        lambda module, action, values: ((profile,), None) if action == "ModelsLoader" else ((), None),
    )
    graph = modular_graph(profile.model_type, "text_to_image")
    graph["nodes"]["loader"]["params"]["mode"] = {"value": "text_to_image"}
    graph["nodes"]["denoise"]["params"]["mode"] = {"value": "image_to_image"}

    def recipe(*args, **kwargs):
        raise AssertionError("Conflicting tasks must not reach recipe selection")

    result = planner.build_workflow_auto_plan(
        graph,
        runtime_fingerprint={},
        local_models=[],
        data_dir=str(tmp_path),
        plan_recipe=recipe,
        hardware={"systemMemory": {}, "accelerator": {}, "offloadDisk": {}},
    )
    assert not result["canAutoRun"] and any("ambiguous" in issue for issue in result["issues"])


def test_ambiguous_declared_tasks_missing_owner_capture_and_unrelated_nodes_stay_explicit(monkeypatch):
    from modiff import workflow_task_identity as identity

    pipeline = "StableDiffusionXLModularPipeline"
    nodes = modular_graph(pipeline, "text_to_image")["nodes"]
    adapters = identity.modular_task_adapters

    def ambiguous(pipeline_class, workflow):
        result = adapters(pipeline_class, workflow)
        return result + [("other_task", adapter) for task, adapter in result if task == "text_to_image"]

    monkeypatch.setattr(identity, "modular_task_adapters", ambiguous)
    assert identity.modular_graph_tasks(nodes, "loader", list(nodes)[1:], pipeline) == {"text_to_image", "other_task"}
    records = {"loader": {"fields": {"model_type": {"value": pipeline}}}}
    assert identity.graph_task_receipts(nodes, records)[0]["task"] is None
    assert identity.graph_task_receipts(nodes, {}) == []
    nodes["unrelated"] = {"module": "custom.Developer", "action": "Process", "params": {}}
    assert identity.resource_consumers(nodes, "loader", {"loader"}) == ["decoder", "denoise", "text_encoder"]


def test_receipts_use_only_the_output_ancestry_and_leave_incomplete_graphs_unlabeled():
    from modiff.execution_input_provenance import capture_generation_inputs, build_resolved_execution_inputs

    graph = modular_graph("StableDiffusionXLModularPipeline", "image_to_image")
    graph["nodes"]["other_loader"] = deepcopy(graph["nodes"]["loader"])
    records = {
        key: capture_generation_inputs(
            key, node, {field: p.get("value") for field, p in node["params"].items() if "value" in p}
        )
        for key, node in graph["nodes"].items()
    }
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=0, node_id="decoder")
    assert [row["loaderId"] for row in receipt["graphTasks"]] == ["loader"]
    del records["vae_encoder"]
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=0, node_id="decoder")
    assert "graphTasks" not in receipt


@pytest.mark.parametrize(
    "pipeline,repository,task,resource_mode",
    [
        (
            "StableDiffusionXLModularPipeline",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "image_to_image",
            "edit_image",
        ),
        (
            "StableDiffusionXLModularPipeline",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "text_to_image",
            "text_to_image",
        ),
        ("FluxModularPipeline", "black-forest-labs/FLUX.1-dev", "image_to_image", "image_to_image"),
        ("QwenImageModularPipeline", "Qwen/Qwen-Image-2512", "text_to_image", "modular_text_to_image"),
    ],
)
def test_actual_profiles_use_their_existing_resource_mode_alias(tmp_path, pipeline, repository, task, resource_mode):
    graph = modular_graph(pipeline, task)
    graph["nodes"]["loader"]["params"]["repo_id"] = {"value": {"source": "hub", "value": repository}}
    requests = []

    def recipe(payload, **kwargs):
        requests.append(payload)
        return {"candidates": []}

    result = build_workflow_auto_plan(
        graph,
        runtime_fingerprint={},
        local_models=[],
        data_dir=str(tmp_path),
        plan_recipe=recipe,
        hardware={"systemMemory": {}, "accelerator": {}, "offloadDisk": {}},
    )
    assert requests and requests[0]["form"]["mode"] == resource_mode
    assert not result["canAutoRun"], "The existing resource profile still needs a valid recipe."


def test_every_published_modular_starter_recognizes_its_task_without_claiming_unique_aliases():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    from modiff.workflow_task_identity import modular_graph_tasks, resource_consumers

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    selections = sorted({(c['pipelineClass'], c['task']) for c in contracts
                         if c['task'] and c['nodeKey'].startswith('modules.ModularDiffusers.')})
    failures = []
    for pipeline, task in selections:
        starter = resolve_operation_starter(MODULE_MAP, contracts, {'pipelineClass': pipeline, 'task': task})
        nodes = {
            n['operation']['operationId']: {
                'module': n['module'], 'action': n['action'],
                'params': {k: {'value': p.get('value', p.get('default'))}
                           for k, p in n['params'].items() if p.get('display') != 'output'},
            } for n in starter['nodes']
        }
        for edge in starter['edges']:
            nodes[edge['target']]['params'][edge['targetHandle']] = {
                'sourceId': edge['source'], 'sourceKey': edge['sourceHandle'],
            }
        for required in starter['requiredInputs']:
            nodes[required['operationId']]['params'][required['field']] = {
                'sourceId': 'external', 'sourceKey': required['field'],
            }
        nodes['external'] = {'module': 'custom.Source', 'action': 'Values', 'params': {}}
        loader = starter['nodes'][0]['operation']['operationId']
        before = deepcopy(nodes)
        actual = modular_graph_tasks(nodes, loader, resource_consumers(nodes, loader, {loader}), pipeline)
        if task not in actual:
            failures.append((pipeline, task, actual))
        assert nodes == before
    assert not failures, repr(failures)


def test_explicit_upstream_composition_uses_validated_placements_and_rejects_partial_or_forged_scope():
    from modiff.huggingface_node_library import reviewed_huggingface_node_library
    from modiff.workflow_task_identity import modular_graph_tasks

    library = reviewed_huggingface_node_library()
    definition = next(d for d in library['definitions'] if d.get('pipelineClass') == 'QwenImageModularPipeline'
                      and d.get('workflowId') == 'text2image')
    blocks = {b['id']: b for b in library['blockDefinitions']}
    nodes = {'loader': {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader', 'params': {}}}
    for i, placement in enumerate(definition['blockPlacements']):
        block = blocks[placement['blockDefinitionId']]
        if block['kind'] not in ('block', 'loop'):
            continue
        values = {'pipeline_class': 'QwenImageModularPipeline', 'workflow_id': 'text2image',
                  'placement_path': placement['path'], 'block_definition_id': block['id'],
                  'block_class': block['className'], 'block_contract_hash': block['contentHash']}
        nodes[f'step-{i}'] = {'module': 'modules.ModularDiffusers', 'action': 'ReviewedModularWorkflowStep',
                             'params': {k: {'value': v} for k, v in values.items()}}
    before = deepcopy(nodes)
    assert modular_graph_tasks(nodes, 'loader', list(nodes)[1:], 'QwenImageModularPipeline') == {'text_to_image'}
    assert nodes == before
    partial = [k for k, n in nodes.items() if k != 'loader' and 'Text' in n['params']['block_class']['value']]
    assert partial
    assert modular_graph_tasks(nodes, 'loader', partial, 'QwenImageModularPipeline') == set()
    first = next(k for k in nodes if k != 'loader')
    nodes[first]['params']['block_contract_hash']['value'] = 'sha256:' + '0' * 64
    assert modular_graph_tasks(nodes, 'loader', list(nodes)[1:], 'QwenImageModularPipeline') == set()


@pytest.mark.parametrize('pipeline', ['Flux3ModularPipeline', 'Flux3DistilledModularPipeline'])
def test_source_wrapper_transports_match_only_the_complete_declared_bundle(pipeline):
    from modiff.workflow_task_identity import modular_graph_tasks

    nodes = modular_graph(pipeline, 'edit_image')['nodes']
    nodes['loader']['params']['workflow_id'] = {'value': 'image_conditioned'}
    consumers = list(nodes)[1:]
    # Existing upstream-only topology stays recognizable; this grants no
    # runtime/provider authority and does not mutate or repair saved graphs.
    assert modular_graph_tasks(nodes, 'loader', consumers, pipeline) == {'edit_image'}
    for target in ('text_encoder', 'denoise'):
        nodes[target]['params']['reference_binding'] = {
            'sourceId': 'vae_encoder', 'sourceKey': 'reference_binding',
        }
    before = deepcopy(nodes)
    assert modular_graph_tasks(nodes, 'loader', consumers, pipeline) == {'edit_image'}
    assert nodes == before
    for target in ('text_encoder', 'denoise'):
        partial = deepcopy(nodes)
        del partial[target]['params']['reference_binding']
        assert modular_graph_tasks(partial, 'loader', consumers, pipeline) == set()
    forged = deepcopy(nodes)
    forged['denoise']['params']['reference_binding']['sourceKey'] = 'unreviewed_binding'
    assert modular_graph_tasks(forged, 'loader', consumers, pipeline) == set()
    extra = deepcopy(nodes)
    extra['denoise']['params']['unreviewed_transport'] = {
        'sourceId': 'vae_encoder', 'sourceKey': 'reference_binding',
    }
    assert modular_graph_tasks(extra, 'loader', consumers, pipeline) == set()
