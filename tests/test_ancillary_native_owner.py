"""Native Block turnover: real serialized dispatcher and tiny official models.

The loader factory is deliberately unit-installed, not pretrained acquisition or
memory qualification. All stage execution and resource ownership use the app.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import weakref

import pytest

import test_ancillary_execution as ancillary
import test_ancillary_owner_lifecycle as lifecycle
import test_flux3_tiny_stage_dispatch as stage_dispatch

actual_pup = lifecycle.actual_pup
tiny_stage_owner = stage_dispatch.tiny_stage_owner
isolated_manager = lifecycle.isolated_manager


@pytest.mark.parametrize('entry,registry,accepted', [
    ({'display': 'input'}, {'display': 'input', 'value': None}, True),
    ({'display': 'input', 'value': False}, {'display': 'input', 'value': None}, False),
    ({'display': 'input', 'value': 0}, {'display': 'input', 'value': None}, False),
    ({'display': 'input', 'value': ''}, {'display': 'input', 'value': None}, False),
    ({'display': 'input', 'value': None}, {'display': 'input', 'value': None}, False),
    ({'display': 'input', 'sourceId': 'other', 'sourceKey': 'output'}, {'display': 'input', 'value': None}, False),
    ({'display': 'input', 'sourceId': None}, {'display': 'input', 'value': None}, False),
    ({'display': 'input'}, {'display': 'input', 'value': False}, False),
    ({'display': 'input'}, {'display': 'input', 'value': 0}, False),
    ({'display': 'input'}, {'display': 'input', 'default': False}, False),
    ({'display': 'input'}, {'display': 'input', 'default': 0}, False),
    ({'display': 'input'}, {'display': 'input', 'value': None, 'required': True}, False),
    ({'display': 'input'}, {'display': 'output', 'value': None}, False),
    ({'display': 'input'}, None, False),
])
def test_selected_native_role_accepts_only_inert_optional_registry_socket(entry, registry, accepted):
    """Normal native export retains an unset socket omitted by selected specs."""
    from modiff.ancillary_execution import AncillaryExecutionError, check_node
    params = {'used': {'display': 'input', 'value': 'selected'}}
    generic = deepcopy(params)
    if registry is not None:
        generic['unused'] = registry
    modules = {'modules.Native': {'Stage': {'params': generic}}}
    nodes = {'stage': {'module': 'modules.Native', 'action': 'Stage',
                      'params': {'used': {'value': 'selected'}, 'unused': entry}}}
    before = deepcopy(nodes)
    if accepted:
        assert check_node(nodes, 'stage', 'modules.Native.Stage', modules, params) is nodes['stage']
    else:
        with pytest.raises(AncillaryExecutionError, match='undeclared'):
            check_node(nodes, 'stage', 'modules.Native.Stage', modules, params)
    assert nodes == before


@pytest.mark.parametrize('entry,default,hidden,accepted', [
    ({'value': 'native'}, 'native', True, True),
    ({'value': 'whole_v1'}, 'native', True, False),
    ({'value': False}, 'native', True, False),
    ({'value': 0}, 'native', True, False),
    ({'value': None}, 'native', True, False),
    ({'value': 'native', 'sourceId': None}, 'native', True, False),
    ({'sourceId': 'other', 'sourceKey': 'output'}, 'native', True, False),
    ({'value': 'native'}, 'whole_v1', True, False),
    ({'value': 'native'}, 'native', False, False),
])
def test_selected_loader_allows_only_unchanged_legacy_native_compatibility(entry, default, hidden, accepted):
    from modiff.ancillary_execution import AncillaryExecutionError, check_node
    registry = {'inpaint_compatibility': {'type': 'string', 'default': default, 'hidden': hidden}}
    modules = {'modules.ModularDiffusers': {'ModelsLoader': {'params': registry}}}
    nodes = {'stage': {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader',
                      'params': {'inpaint_compatibility': entry}}}
    before = deepcopy(nodes)
    if accepted:
        assert check_node(nodes, 'stage', 'modules.ModularDiffusers.ModelsLoader', modules, {}) is nodes['stage']
    else:
        with pytest.raises(AncillaryExecutionError, match='undeclared'):
            check_node(nodes, 'stage', 'modules.ModularDiffusers.ModelsLoader', modules, {})
    assert nodes == before


def native_document(graph, modules, root='image-block', *, lora_siblings=False):
    """Build through the existing strict definition/instance contract producers."""
    from modiff.block_definition_v2 import (
        block_definition_content_hash_v2, block_graph_hash_v2,
        block_interface_hash_v2, validate_block_definition_v2,
        validate_block_instance_v2,
    )
    from test_composite_migration_inventory import v2_definition
    value = deepcopy(graph)
    binding = value['runtimeHints']['executionPurpose']['bindings'][0]
    from modiff.studio_execution_specs import (
        STUDIO_EXECUTION_SPEC_DEFINITIONS, _public_spec, _execution_spec_role_params)
    execution_spec = _public_spec(binding['parent']['baseSpecId'],
                        STUDIO_EXECUTION_SPEC_DEFINITIONS[binding['parent']['baseSpecId']])
    ids = binding['parentRoleNodes']
    data_ids = {f'aux-lora-{index}': node_id
        for index, (node_id, api) in enumerate(value['nodes'].items())
        if (api['module'], api['action']) == ('modules.ModularDiffusers', 'Lora')}
    native_ids = {**ids, **({} if lora_siblings else data_ids)}
    projected = {node_id: f'block-v2-node:{len(root)}:{root}:{len(role)}:{role}'
                 for role, node_id in native_ids.items()}
    nodes = []
    for role, node_id in native_ids.items():
        api = value['nodes'][node_id]
        params = deepcopy(_execution_spec_role_params(execution_spec,
            api['module'] + '.' + api['action'], modules[api['module']][api['action']]))
        params = object.__new__(lifecycle.WebServer).describe_node_params(params)
        for field, spec in api['params'].items():
            if 'value' in spec:
                params[field]['value'] = deepcopy(spec['value'])
        # The real HTTP JSON boundary normalizes tuple/list and object keys.
        params = json.loads(json.dumps(params))
        nodes.append({'nodeId': role, 'nodeType': 'custom', 'semanticRole': role,
                      'data': {'type': 'custom', 'module': api['module'],
                               'action': api['action'], 'label': role, 'params': params}})
    edges = []
    roles = {node_id: role for role, node_id in native_ids.items()}
    for target, api in value['nodes'].items():
        for field, spec in api['params'].items():
            if spec.get('sourceId') in roles and target in roles:
                edges.append({'edgeId': f'edge-{len(edges)}',
                              'sourceNodeId': roles[spec['sourceId']], 'sourcePortId': spec['sourceKey'],
                              'targetNodeId': roles[target], 'targetPortId': field})
    definition = v2_definition()
    definition['definitionId'] = 'unit-native-image-stages'
    definition['graph'] = {'nodes': nodes, 'edges': edges,
                           'executionOrder': [roles[node_id] for path in value['paths'] for node_id in path if node_id in roles],
                           'graphHash': 'pending'}
    definition['graph']['graphHash'] = block_graph_hash_v2(definition['graph'])
    definition['boundary'] = {'mode': 'explicit', 'inputs': [], 'outputs': []}
    if lora_siblings:
        definition['boundary']['inputs'] = [{'portId': 'models:lora_list', 'label': 'LoRA',
            'valueType': 'custom_lora', 'required': False,
            'binding': {'nodeId': 'models', 'fieldOrPortId': 'lora_list'}}]
    definition['controls'] = []
    definition['suggestedInputs'] = []
    definition['previews'] = []
    definition['contentHash'] = block_definition_content_hash_v2(definition)
    validate_block_definition_v2(definition)
    interface_hash = block_interface_hash_v2(definition)
    instance = {'schemaVersion': 2, 'instanceId': root,
        'definitionRef': {'definitionId': definition['definitionId'], 'contentHash': definition['contentHash']},
        'definitionSnapshot': definition, 'effectiveGraph': deepcopy(definition['graph']),
        'effectiveInterface': {'boundary': deepcopy(definition['boundary']), 'controls': [],
                              'baseInterfaceHash': interface_hash, 'effectiveInterfaceHash': interface_hash},
        'values': {}, 'customization': {'state': 'unchanged',
            'baseGraphHash': definition['graph']['graphHash'], 'effectiveGraphHash': definition['graph']['graphHash']},
        'presentation': {'expanded': False, 'position': {'x': 0, 'y': 0},
                         'size': {'width': 400, 'height': 400}, 'internalLayout': {}},
        'previewStates': [], 'authorities': []}
    validate_block_instance_v2(instance)
    value['nodes'] = {projected.get(node_id, node_id): api for node_id, api in value['nodes'].items()}
    for api in value['nodes'].values():
        for spec in api['params'].values():
            if spec.get('sourceId') in projected:
                spec['sourceId'] = projected[spec['sourceId']]
    value['paths'] = [[projected.get(node_id, node_id) for node_id in path] for path in value['paths']]
    for name in ('parentRoleNodes',):
        binding[name] = {role: projected[node_id] for role, node_id in binding[name].items()}
    value['runtimeHints']['studioExecutionSpec']['nodes'] = dict(binding['parentRoleNodes'])
    value['runtimeHints']['workflowSnapshot'] = {'nodes': [{'id': root, 'type': 'block',
        'data': {'blockInstanceV2': instance}, 'position': {'x': 0, 'y': 0}}], 'edges': []}
    if lora_siblings:
        snapshot = value['runtimeHints']['workflowSnapshot']
        for node_id in data_ids.values():
            api = value['nodes'][node_id]
            snapshot['nodes'].append({'id': node_id, 'type': 'custom',
                'data': {'module': api['module'], 'action': api['action']}, 'position': {'x': 0, 'y': 0}})
        snapshot['edges'].append({'id': 'unit-native-lora-boundary',
            'source': next(iter(data_ids.values())), 'sourceHandle': 'lora',
            'target': root, 'targetHandle': 'models:lora_list'})
    return value


def test_ordinary_private_starter_capture(tmp_path, monkeypatch):
    """Real DTO producer with constructor traps; public refusal stays intact."""
    import modules
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    from modiff.server import WebServer
    from modiff.NodeBase import NodeBase
    contracts, _ = build_operation_catalog(modules.MODULE_MAP, [], catalog_resolver=lambda: {})
    from modiff.ancillary_execution import PRIVATE_PARENTS
    identity, parent = next(iter(PRIVATE_PARENTS.items()))
    selection = {'pipelineClass': parent['pipelineClass'], 'task': parent['task'],
                 'executionProfileId': parent['profileId'], 'authoringRecipeId': identity}
    monkeypatch.setattr(NodeBase, '__init__', lambda *_args, **_kwargs: pytest.fail('metadata constructed a node'))
    with pytest.raises(ValueError, match='execution profile'):
        resolve_operation_starter(modules.MODULE_MAP, contracts, selection)
    # Capture the actual ordinary capability producer before the private
    # starter-only fixture override. Its contract-only eligibility stays closed.
    capability_app = WebServer(modules.MODULE_MAP, work_dir=str(tmp_path), data_dir=str(tmp_path))
    public_payload = capability_app._build_model_capabilities_payload('flux3')
    selected_capability = next(item for item in public_payload['capabilities']
        if item['modelType'] == parent['pipelineClass'])
    assert selected_capability['autoEligible'] is False and selected_capability['templateEligible'] is False
    profile = DIFFUSERS_EXECUTION_PROFILES[parent['profileId']]
    monkeypatch.setitem(DIFFUSERS_EXECUTION_PROFILES, profile.id, replace(profile, public=True))
    payload = resolve_operation_starter(modules.MODULE_MAP, contracts, selection)
    app = object.__new__(WebServer)
    payload['nodes'] = [{'operation': node['operation'],
        'node': app._describe_registered_node(node['module'], node['action'], node)} for node in payload['nodes']]
    assert payload['authoring']['ancillaryOperations']
    child = payload['authoring']['ancillaryOperations'][0]
    verification = child.get('verification')
    assert verification is not None, 'ordinary metadata must independently declare proposal verification expectations'
    assert verification['adapter'] == child['adapter']
    assert set(verification['systemPromptSha256ByMode']) == {'t2i', 'i2i', 't2i_bbox', 'i2i_bbox'}
    assert verification['systemPromptsSha256'].startswith('sha256:')
    assert verification['chatTemplateSha256'].startswith('sha256:')
    assert not {'device', 'dtype', 'quantization'} & set(verification['modelReceipt']['runtime'])
    path = Path(__import__('os').environ['TEMP']).parent / 'ordinary-private-starter-capture.json'
    path.write_text(json.dumps({'proofScope': 'Constructor-trapped ordinary metadata producer; private profile selection override only; no publication/admission.',
        'selection': selection, 'response': payload, 'operationContracts': contracts,
        'selectedModelCapability': selected_capability,
        'ancillaryNodeRegistry': {key: app._describe_registered_node(*key.rsplit('.', 1),
            modules.MODULE_MAP[key.rsplit('.', 1)[0]][key.rsplit('.', 1)[1]])
            for _role, key, _x, _y in child['roles']},
        'selectedModelCapabilityProvenance': 'Actual ordinary _build_model_capabilities_payload before the private starter fixture override; eligibility remains contract-only.'}, indent=2) + '\n', encoding='utf-8')
    assert path.is_file()


def test_native_document_fixture_preserves_existing_graph_validation():
    import modules
    from modiff.ancillary_execution import validate_execution_purpose
    from modiff.studio_execution_specs import assert_studio_execution_graph
    graph = native_document(ancillary.workflow_graph(modules.MODULE_MAP), modules.MODULE_MAP)
    validate_execution_purpose(graph, modules.MODULE_MAP)
    assert_studio_execution_graph(graph, graph['runtimeHints'])


@pytest.mark.parametrize('fault', ['none', 'presentation', 'duplicate-root', 'projected-collision',
    'ordinary-class', 'duplicate-role', 'wrong-boundary', 'missing-edge', 'extra-parent-edge',
    'foreign-supplier-edge', 'flat-wire'])
def test_actual_client_split_stage_metadata_has_native_association(fault):
    """Real factory/group/export context only; not dispatcher or model proof."""
    import modules
    from modiff.ancillary_execution import native_parent_association
    witness = json.loads((Path(__file__).parent / 'fixtures/flux3-split-stages-client.v1.json').read_text())
    graph = deepcopy(witness['apiGraph'])
    parent = ancillary.parent_for(modules.MODULE_MAP)
    child = parent['ancillaryOperations'][0]
    binding = graph['runtimeHints']['executionPurpose']['bindings'][0]
    # This fixture is a fresh real client export of the current producer.
    # Never silently rebind an old capture into structural acceptance here.
    assert binding['parent'] == {**child['parent'], 'authoringContentHash': parent['contentHash']}
    assert binding['contentHash'] == child['contentHash']
    snapshot = graph['runtimeHints']['workflowSnapshot']
    ids = binding['parentRoleNodes']
    if fault == 'presentation':
        snapshot['nodes'][0]['position'] = {'x': 999, 'y': -999}
    elif fault == 'duplicate-root':
        snapshot['nodes'].append(deepcopy(snapshot['nodes'][0]))
    elif fault == 'projected-collision':
        extra = deepcopy(snapshot['nodes'][0])
        extra['id'] = ids['models']
        snapshot['nodes'].append(extra)
    elif fault == 'ordinary-class':
        next(node for node in snapshot['nodes'] if node['id'] == ids['denoise'])['data']['action'] = 'CustomDenoise'
    elif fault == 'duplicate-role':
        binding['parentRoleNodes']['prompt'] = ids['models']
    elif fault == 'wrong-boundary':
        snapshot['edges'][0]['sourceHandle'] = 'not-a-declared-port'
    elif fault == 'missing-edge':
        snapshot['edges'].pop(0)
    elif fault == 'extra-parent-edge':
        extra = deepcopy(snapshot['edges'][0])
        extra['id'] = 'extra-parent-wire'
        extra['target'] = ids['denoise']
        extra['targetHandle'] = 'extra'
        snapshot['edges'].append(extra)
    elif fault == 'foreign-supplier-edge':
        snapshot['edges'].append({'id': 'foreign-supplier-wire', 'source': 'foreign-owner',
            'sourceHandle': 'model', 'target': ids['denoise'], 'targetHandle': 'extra'})
    elif fault == 'flat-wire':
        graph['nodes'][ids['denoise']]['params']['unet']['sourceKey'] = 'vae_out'
    before = deepcopy(graph)
    result = native_parent_association(graph, binding, modules.MODULE_MAP)
    assert (result is not None) == (fault in {'none', 'presentation'})
    assert graph == before


def test_verification_metadata_refuses_changed_system_prompt_bundle(tmp_path, monkeypatch):
    import modules
    from modiff import prompt_enhancement
    from modiff.NodeBase import NodeBase
    monkeypatch.setattr(NodeBase, '__init__', lambda *_args, **_kwargs: pytest.fail('metadata constructed a node'))
    changed = tmp_path / 'changed-system-prompts.json'
    changed.write_bytes(prompt_enhancement.SYSTEM_PROMPTS.read_bytes() + b'\n')
    monkeypatch.setattr(prompt_enhancement, 'SYSTEM_PROMPTS', changed)
    with pytest.raises(ValueError, match='system-prompt verification metadata'):
        ancillary.parent_for(modules.MODULE_MAP)


@pytest.mark.parametrize('fault', ['none', 'client-split', 'client-i2i-split', 'presentation', 'parent-key-order', 'resource-snapshot', 'inactive-route', 'missing-origin', 'changed-inputs',
    'changed-block', 'shared-component', 'active-owner', 'foreign-cache', 'source-drift',
    'surviving-reference', 'failed-image', 'input-bound', 'connected-input-bound', 'input-drift',
    'preparation-ram', 'post-release-ram',
    'lora-data', 'lora-chain', 'lora-data-sibling', 'lora-sibling-wire-changed',
    'lora-data-changed', 'lora-code-changed', 'lora-input-changed'])
def test_completed_native_block_owner_turns_over_before_actual_pup_factory(
        actual_pup, tiny_stage_owner, tmp_path, monkeypatch, fault):
    """Actual stages/dispatch/release; tiny factory substitution is explicit."""
    import modules
    from modules.ModularDiffusers.loaders import ModelsLoader
    from modules.ModularDiffusers.route_state import bind_loader_outputs, issue_pipeline_instance_token
    from modiff.NodeBase import NodeBase, memory_manager

    client_split = fault in {'client-split', 'client-i2i-split'}
    image_conditioned = fault == 'client-i2i-split'
    memory_manager.remove(actual_pup[0]['model'])
    if client_split:
        filename = 'flux3-i2i-split-stages-client.v1.json' if image_conditioned else 'flux3-split-stages-client.v1.json'
        witness = json.loads((Path(__file__).parent / 'fixtures' / filename).read_text())
        graph = deepcopy(witness['apiGraph'])
        assert witness['binding'] == graph['runtimeHints']['executionPurpose']['bindings'][0]
        # actual_pup deliberately substitutes tiny source weights and its
        # descriptor through real serializers. Renew only that declaration
        # binding through the actual producer; preserve the exact captured
        # client grouping, role IDs, graph, ports and document snapshot.
        fresh_parent = ancillary.parent_for(modules.MODULE_MAP, 1 if image_conditioned else 0)
        fresh_child = fresh_parent['ancillaryOperations'][0]
        selected_binding = graph['runtimeHints']['executionPurpose']['bindings'][0]
        selected_binding['parent'] = {**fresh_child['parent'], 'authoringContentHash': fresh_parent['contentHash']}
        selected_binding['contentHash'] = fresh_child['contentHash']
        # Same immutable upstream index fixture used by the maintained native
        # LoRA runtime tests. Only acquisition metadata is installed here; the
        # neural factory remains the explicit tiny substitution below.
        from modiff.config import CONFIG
        selected = graph['nodes'][selected_binding['parentRoleNodes']['models']]['params']
        repository = selected['repo_id']['value']['value']
        revision = selected['revision']['value']
        snapshot = Path(CONFIG.hf['cache_dir']) / ('models--' + repository.replace('/', '--')) / 'snapshots' / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        index = Path(__file__).parent / 'fixtures' / (repository.rsplit('/', 1)[1] + '.modular-index.v1.json')
        (snapshot / 'modular_model_index.json').write_bytes(index.read_bytes())
        from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS, _public_spec, _execution_spec_role_params
        from modiff.ancillary_execution import check_node
        public = _public_spec(fresh_parent['baseSpecId'], STUDIO_EXECUTION_SPEC_DEFINITIONS[fresh_parent['baseSpecId']])
        for role, key, _x, _y in fresh_parent['roles']:
            api = graph['nodes'][selected_binding['parentRoleNodes'][role]]
            generic = modules.MODULE_MAP[api['module']][api['action']]['params']
            schema = _execution_spec_role_params(public, key, {'params': generic})
            try:
                check_node(graph['nodes'], selected_binding['parentRoleNodes'][role], key, modules.MODULE_MAP, schema)
            except ValueError as error:
                raise AssertionError({name: {'api': value, 'registry': generic.get(name)}
                    for name, value in api['params'].items() if name not in schema}) from error
    else:
        graph = ancillary.workflow_graph(modules.MODULE_MAP)
    if image_conditioned:
        # The real exporter authors relative identifiers only. Install explicit
        # synthetic unit media in that configured root without changing them.
        from PIL import Image
        from modiff.config import CONFIG
        media_root = Path(CONFIG.paths['work_dir']) / 'unit-media'
        media_root.mkdir(parents=True, exist_ok=True)
        for filename, color in [('reference-a.png', 'red'), ('reference-b.png', 'blue')]:
            Image.new('RGB', (64, 64), color).save(media_root / filename)
    ids = graph['runtimeHints']['executionPurpose']['bindings'][0]['parentRoleNodes']
    for api in graph['nodes'].values():
        schema = modules.MODULE_MAP[api['module']][api['action']]['params']
        for field, value in api['params'].items():
            if value == {'value': None} and field in schema and 'value' in schema[field]:
                value['value'] = deepcopy(schema[field]['value'])
    graph['nodes'][ids['prompt']]['params']['max_sequence_length'] = {'value': 8}
    graph['nodes'][ids['denoise']]['params'].update({
        'width': {'value': 64}, 'height': {'value': 64},
        'num_inference_steps': {'value': 2}, 'num_images_per_prompt': {'value': 1}})
    graph['runtimeHints']['resourceMode'] = 'expert'
    if fault in {'lora-data', 'lora-chain', 'lora-data-sibling', 'lora-sibling-wire-changed',
                 'lora-data-changed', 'lora-code-changed', 'lora-input-changed'}:
        import torch
        from safetensors.torch import save_file
        path = tmp_path / 'models/native-image-lora.safetensors'
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({'transformer.mode_blocks.image.0.attn.to_q.lora_A.weight': torch.ones(2, 16),
                   'transformer.mode_blocks.image.0.attn.to_q.lora_B.weight': torch.ones(16, 2)}, str(path))
        graph['nodes']['image-lora-data'] = {'module': 'modules.ModularDiffusers', 'action': 'Lora',
            'params': {name: {'value': value} for name, value in {
                'model': {'source': 'local', 'value': str(path)}, 'scale': 1.0,
                'adapter_name': 'native-turnover-unit', 'weight_name': path.name}.items()}}
        graph['nodes'][ids['models']]['params']['lora_list'] = {'sourceId': 'image-lora-data', 'sourceKey': 'lora'}
        graph['paths'][0].insert(0, 'image-lora-data')
        if fault == 'lora-chain':
            graph['nodes']['image-lora-earlier'] = deepcopy(graph['nodes']['image-lora-data'])
            graph['nodes']['image-lora-earlier']['params']['adapter_name'] = {'value': 'native-turnover-earlier'}
            graph['nodes']['image-lora-data']['params']['previous_loras'] = {'sourceId': 'image-lora-earlier', 'sourceKey': 'lora'}
            graph['paths'][0].insert(0, 'image-lora-earlier')
    if not client_split:
        graph = native_document(graph, modules.MODULE_MAP,
            lora_siblings=fault in {'lora-data-sibling', 'lora-sibling-wire-changed'})
    if fault == 'inactive-route':
        instance = graph['runtimeHints']['workflowSnapshot']['nodes'][0]['data']['blockInstanceV2']
        instance['routeSelection'] = {'schemaVersion': 1, 'routeSetId': 'diffusers.definition-switch:v1',
                                      'selectedRouteKey': 'image-current', 'inactiveDrafts': {}}
    ids = graph['runtimeHints']['executionPurpose']['bindings'][0]['parentRoleNodes']
    model_refs = {}

    def tiny_factory(node, **_kwargs):
        from diffusers import FlowMatchEulerDiscreteScheduler
        owner = tiny_stage_owner('Flux3DistilledModularPipeline')
        manager = owner.manager
        old_ids = manager.collections.pop('unit-tiny-owner')
        clones = {'text_encoder': deepcopy(owner.encoder), 'tokenizer': deepcopy(owner.processor),
                  'vae': deepcopy(owner.vae), 'transformer': deepcopy(owner.transformer),
                  'scheduler': FlowMatchEulerDiscreteScheduler(shift=6.93)}
        for key in old_ids:
            manager.components.pop(key)
        component_ids = {name: manager.add(name, component, collection=node.node_id)
                         for name, component in clones.items()}
        model_refs.update({name: weakref.ref(component) for name, component in clones.items()
                           if hasattr(component, 'parameters')})
        common = {'model_type': 'Flux3DistilledModularPipeline', 'repo_id': 'owned-local-tiny-components',
                  'repo_source': 'local', 'revision': None}
        token = issue_pipeline_instance_token(**common)
        return bind_loader_outputs({
            'text_encoders': {**common, 'text_encoder': {'model_id': component_ids['text_encoder']},
                              'tokenizer': {'model_id': component_ids['tokenizer']}},
            'unet_out': {**common, 'model_id': component_ids['transformer']},
            'scheduler': {**common, 'model_id': component_ids['scheduler']},
            'vae_out': {**common, 'model_id': component_ids['vae']}}, token)

    monkeypatch.setattr(ModelsLoader, 'execute', tiny_factory)
    # Isolate acquisition and its pretrained-cache preflight together. The
    # actual native class still uses NodeBase, the dispatcher, real stages,
    # actual ComponentsManager ownership, and ordinary release. This is not
    # evidence for the production ModelsLoader acquisition contract.
    monkeypatch.setattr(ModelsLoader, '__call__', NodeBase.__call__)
    app = lifecycle.app_for(tmp_path, monkeypatch, graph)
    app._prepare_auto_runtime_for_graph = lambda _hints: None
    app._best_effort_device_cache_clear = lambda: []
    app._best_effort_allocator_trim = lambda: (False, [])
    def fail_completed_image(node_id):
        if fault == 'failed-image' and node_id == ids['preview']:
            raise ValueError('unit image stage failed before terminal sealing')
    lifecycle.quiet_dispatch(app, fail_completed_image)
    if fault == 'failed-image':
        with pytest.raises(ValueError, match='unit image stage failed'):
            lifecycle.leased(app, graph)
    else:
        lifecycle.leased(app, graph)
    assert model_refs and all(reference() is not None for reference in model_refs.values())
    assert app.node_cache[ids['decode']].output['images']

    if image_conditioned:
        preview = deepcopy(witness['previewApiGraph'])
        binding = preview['runtimeHints']['executionPurpose']['bindings'][0]
        binding['parent'] = {**fresh_child['parent'], 'authoringContentHash': fresh_parent['contentHash']}
        binding['contentHash'] = fresh_child['contentHash']
    else:
        preview, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    preview['runtimeHints'].update(resourceMode='expert', workflowSnapshot=deepcopy(graph['runtimeHints']['workflowSnapshot']))
    preview['runtimeHints']['executionPurpose']['bindings'][0]['parentRoleNodes'] = dict(ids)
    incoming = preview['runtimeHints']['executionPurpose']['bindings'][0]['roleNodes']
    preview['nodes'][incoming['provider']]['params']['device'] = {'value': lifecycle.pup.DEFAULT_DEVICE}
    preview['nodes'][incoming['proposal']]['params'].update(enhancement_max_new_tokens={'value': 2}, min_new_tokens={'value': 2})
    if fault == 'client-split':
        children = graph['runtimeHints']['executionPurpose']['bindings'][0]['roleNodes']
        preview['nodes'] = {children.get(node_id, node_id): api for node_id, api in preview['nodes'].items()}
        for api in preview['nodes'].values():
            for param in api['params'].values():
                if param.get('sourceId') in children:
                    param['sourceId'] = children[param['sourceId']]
        preview['paths'] = [[children.get(node_id, node_id) for node_id in path] for path in preview['paths']]
        preview['runtimeHints']['executionPurpose']['bindings'][0]['roleNodes'] = dict(children)
        preview['runtimeHints']['executionPurpose']['target']['nodeId'] = children['proposal']
    child_ids = preview['runtimeHints']['executionPurpose']['bindings'][0]['roleNodes']
    from transformers import AutoModelForImageTextToText
    actual_load = AutoModelForImageTextToText.from_pretrained
    reached = []
    def require_released_image(*args, **kwargs):
        assert all(reference() is None for reference in model_refs.values()), 'image model still alive at PUP factory'
        reached.append(True)
        if fault == 'input-drift':
            from transformers import Qwen3VLProcessor
            original_processor = Qwen3VLProcessor.__call__
            def changed_inputs(processor, *args, **kwargs):
                batch = original_processor(processor, *args, **kwargs)
                batch['input_ids'] = batch['input_ids'].clone()
                batch['input_ids'][0, -1] = 1
                return batch
            monkeypatch.setattr(Qwen3VLProcessor, '__call__', changed_inputs)
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(AutoModelForImageTextToText, 'from_pretrained', require_released_image)
    held = None
    if fault == 'missing-origin':
        app.__dict__.pop('_ancillary_dispatched_origins', None)
    elif fault == 'changed-inputs':
        app.node_cache[ids['models']]._cache_input_snapshot = {'changed': True}
    elif fault == 'changed-block':
        preview['runtimeHints']['workflowSnapshot']['nodes'][0]['data']['blockInstanceV2']['definitionRef']['contentHash'] = 'block-definition-v2-00000000'
    elif fault == 'presentation':
        preview['runtimeHints']['workflowSnapshot']['nodes'][0]['data']['blockInstanceV2']['presentation']['position']['x'] = 900
    elif fault == 'parent-key-order':
        binding = preview['runtimeHints']['executionPurpose']['bindings'][0]
        binding['parent'] = {key: binding['parent'][key] for key in reversed(binding['parent'])}
    elif fault == 'resource-snapshot':
        from modiff.block_definition_v2 import block_graph_hash_v2
        instance = preview['runtimeHints']['workflowSnapshot']['nodes'][0]['data']['blockInstanceV2']
        model = next(node for node in instance['effectiveGraph']['nodes'] if node['nodeId'] == 'models')
        current = model['data']['params']['dtype'].get('value')
        model['data']['params']['dtype']['value'] = 'float32' if current != 'float32' else 'bfloat16'
        instance['effectiveGraph']['graphHash'] = block_graph_hash_v2(instance['effectiveGraph'])
        instance['customization'].update(state='structure_changed', effectiveGraphHash=instance['effectiveGraph']['graphHash'])
    elif fault == 'inactive-route':
        instance = preview['runtimeHints']['workflowSnapshot']['nodes'][0]['data']['blockInstanceV2']
        instance['routeSelection']['inactiveDrafts']['archived'] = {'schemaVersion': 1,
            'routeKey': 'archived', **{key: deepcopy(instance[key]) for key in (
                'definitionRef', 'definitionSnapshot', 'effectiveGraph', 'effectiveInterface', 'values', 'customization')},
            'internalLayout': {}}
    elif fault == 'shared-component':
        native = __import__('modules.ModularDiffusers', fromlist=['components']).components
        native.collections['unrelated-owner'] = list(native.collections[ids['models']])
    elif fault == 'active-owner':
        memory_manager._active_ids['foreign-active-id'] = 1
    elif fault == 'foreign-cache':
        from types import SimpleNamespace
        app.node_cache['unrelated-owner'] = SimpleNamespace(output={}, params={}, _mm_models=[])
    elif fault == 'source-drift':
        app.backend_source_identity['fingerprint'] = 'sha256:' + '0' * 64
    elif fault == 'surviving-reference':
        held = next(iter(model_refs.values()))()
    elif fault == 'input-bound':
        preview['nodes']['proposal']['params']['max_input_tokens'] = {'value': 1}
    elif fault == 'connected-input-bound':
        preview['nodes']['proposal']['params']['max_input_tokens'] = {'sourceId': 'provider', 'sourceKey': 'receipt'}
    elif fault in {'lora-data-changed', 'lora-code-changed', 'lora-input-changed'}:
        data_id = next(node_id for node_id, api in graph['nodes'].items()
            if (api['module'], api['action']) == ('modules.ModularDiffusers', 'Lora'))
        if fault == 'lora-data-changed':
            app.node_cache[data_id].output['lora']['scale'] = 0.25
        elif fault == 'lora-code-changed':
            monkeypatch.setattr(type(app.node_cache[data_id]), 'execute', lambda *_args, **_kwargs: None)
        else:
            app.node_cache[data_id]._cache_input_snapshot = {'changed': True}
    elif fault == 'lora-sibling-wire-changed':
        preview['runtimeHints']['workflowSnapshot']['edges'][0]['sourceHandle'] = 'receipt'
    if fault in {'preparation-ram', 'post-release-ram'}:
        # Actual completed tiny image owner and serialized dispatcher. Only the
        # incoming budget/runtime policy is controlled: no NF4/CUDA/fit proof.
        from modiff import auto_resource
        from modiff.workflow_auto_resource import workflow_graph_hash
        import transformers
        preview['nodes'][child_ids['provider']]['params'].update(device={'value': 'cuda:0'},
            dtype={'value': 'bfloat16'}, quantization_mode={'value': 'bnb_4bit'})
        preview['nodes'][child_ids['proposal']]['params']['max_input_tokens'] = {'value': 4096}
        graph_hash = workflow_graph_hash(preview)
        preview['runtimeHints'].update(resourceMode='auto',
            workflowAutoPlan={'schemaVersion': 1, 'graphHash': graph_hash})
        plan = {'graphHash': graph_hash, 'plannedGraphHash': graph_hash, 'canAutoRun': True,
            'patches': [], 'schedule': None, 'requiresPreparation': False, 'requiresCachePreparation': False,
            'adapters': [], 'issues': [], 'strategy': 'controlled_auxiliary_owner',
            'requirements': {'systemRamBytes': 6606028800, 'vramBytes': 14274756576, 'diskFreeBytes': 0},
            'loaders': [{'nodeId': child_ids['provider'], 'modelType': 'ImageTextToText',
            'repository': lifecycle.pup.REPOSITORY, 'candidateId': 'controlled-t2i-source-budget',
            'consumers': [child_ids['proposal']], 'capacityRequirements': {},
            'requirements': {'systemRamBytes': 2311061504, 'vramBytes': 12127272928, 'diskFreeBytes': 0},
            'workingMemoryPolicy': 'runtime_headroom_policy',
            'parameterResidencyRequirements': {'systemRamBytes': 0, 'vramBytes': 8552677344},
            'parameterLoadingRequirements': {'systemRamBytes': 2034237440, 'vramBytes': 2034237440},
            'workingDemandEvidence': {'kind': 'reviewed_bounded_image_text_storage',
                'cpuPreparationRequirements': {'systemRamBytes': 268435456, 'vramBytes': 0, 'diskFreeBytes': 0}},
            'settings': {'device': 'cuda:0', 'dtype': 'bfloat16', 'quantizationMode': 'bnb_4bit', 'offloadMode': 'none'}}]}
        app._build_workflow_auto_plan = lambda *_args, **_kwargs: deepcopy(plan)
        capacity_samples = []
        def hardware(*_args):
            capacity_samples.append(len(capacity_samples))
            available = 4 if fault == 'preparation-ram' else 8 if len(capacity_samples) == 1 else 6
            return {'systemMemory': {'totalBytes': 32 * 1024**3, 'availableBytes': available * 1024**3},
                'accelerator': {'kind': 'cuda', 'totalBytes': 16 * 1024**3, 'freeBytes': 16 * 1024**3}}
        monkeypatch.setattr(auto_resource, '_hardware_snapshot', hardware)
        if fault == 'preparation-ram':
            monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained',
                lambda *_args, **_kwargs: pytest.fail('processor constructed before actual preparation RAM gate'))
    app.current_task = {'task_id': 'native-owner-next-preview', 'attempt_index': 0,
                        'args': [preview], 'sid': preview['sid'], 'progress': 0}
    if fault not in {'none', 'client-split', 'client-i2i-split', 'presentation', 'parent-key-order', 'resource-snapshot', 'inactive-route',
                     'lora-data', 'lora-chain', 'lora-data-sibling'}:
        error_type = RuntimeError if fault in {'input-bound', 'surviving-reference', 'input-drift'} else ValueError
        with pytest.raises(error_type, match='(?i)owner|origin|block|component|reference|input.token|processed request|preallocation|actual free memory'):
            if fault == 'post-release-ram':
                # Direct actual serialized preallocation boundary only. This
                # does not bypass optional runtime readiness or claim a CUDA
                # worker: no provider/graph execution or factory is attempted.
                import asyncio
                from modiff.ancillary_owner_lifecycle import owner_scope, bind_preview_auto_plan, before_node
                def boundary():
                    with owner_scope(app, preview, preview['runtimeHints']['executionPurpose']) as _scope:
                        bind_preview_auto_plan(app, plan)
                        before_node(app, child_ids['provider'])
                        app.current_task['current_node'] = child_ids['provider']
                        from modiff.ancillary_owner_lifecycle import before_image_text_factory
                        before_image_text_factory(actual_pup[0]['preprocessor'],
                            selection={'source': 'hub', 'value': lifecycle.pup.REPOSITORY},
                            revision=lifecycle.pup.REVISION)
                async def run_boundary():
                    app.loop = asyncio.get_running_loop()
                    return await app._with_node_cache_lease(
                        lambda: app._run_executor_callback(boundary, serialize_model_io=True))
                try:
                    asyncio.run(run_boundary())
                finally:
                    if app._model_executor is not None:
                        app._model_executor.shutdown(wait=True)
                        app._model_executor = None
            else:
                lifecycle.leased(app, preview)
        if fault == 'input-drift':
            assert reached == [True] and all(reference() is None for reference in model_refs.values())
            assert app.node_cache[child_ids['provider']]._mm_models and memory_manager.cache
            assert app.node_cache[ids['decode']].output['images']
            return
        assert reached == []
        if fault == 'post-release-ram':
            assert capacity_samples == [0, 1]
            assert ids['models'] not in app.node_cache
            assert all(reference() is None for reference in model_refs.values())
            assert app.node_cache[ids['decode']].output['images']
            assert not memory_manager.cache
            return
        if fault == 'preparation-ram':
            assert capacity_samples == [0]
        if fault == 'surviving-reference':
            assert held is not None and ids['models'] not in app.node_cache
            assert app.node_cache[ids['decode']].output['images']
        else:
            assert ids['models'] in app.node_cache
            assert all(reference() is not None for reference in model_refs.values())
        return
    lifecycle.leased(app, preview)
    assert reached == [True]
    summary = json.loads(app.node_cache[child_ids['proposal']].output['proposal_summary'])
    current_parent = ancillary.parent_for(modules.MODULE_MAP)
    expected = current_parent['ancillaryOperations'][0]['verification']
    receipt = summary['modelReceipt']
    for key, value in expected['modelReceipt'].items():
        if key == 'runtime':
            assert {name: receipt[key][name] for name in value} == value
        else:
            assert receipt[key] == value
    assert summary['adapter'] == expected['adapter']
    mode = summary['consumedInput']['fingerprints']['system_prompt_mode']
    assert summary['consumedInput']['fingerprints']['system_prompt_sha256'] == expected['systemPromptSha256ByMode'][mode]
    assert app.node_cache[ids['decode']].output['images']
    assert not memory_manager.cache



def test_private_preview_metadata_binds_actual_effective_controls(monkeypatch):
    import modules
    from modiff.NodeBase import NodeBase
    monkeypatch.setattr(NodeBase, '__init__', lambda *_args, **_kwargs: pytest.fail('metadata constructed a node'))
    child = ancillary.parent_for(modules.MODULE_MAP)['ancillaryOperations'][0]
    verification = child['verification']
    assert verification.get('runtimeInputs') == {'role': 'provider', 'modelField': 'model_id',
        'revisionField': 'revision', 'dtypeField': 'dtype', 'deviceField': 'device',
        'quantizationField': 'quantization_mode'}
    assert verification.get('requestInputs') == {'role': 'proposal', 'operationField': 'operation',
        'maxNewTokensField': 'enhancement_max_new_tokens', 'minNewTokensField': 'min_new_tokens',
        'maxInputTokensField': 'max_input_tokens', 'maxImagePixelsField': 'max_image_pixels'}
    assert verification.get('supportedPromptModes') == ['t2i', 'i2i']
    source = verification['modelReceipt']['source']
    assert ['provider', 'model_id', {'source': 'hub', 'value': source['repository']}] in child['initialValues']
    assert ['provider', 'revision', source['revision']] in child['initialValues']
    assert child.get('fieldOverrides') == [['proposal', 'enhancement_max_new_tokens', {'hidden': False}],
        ['proposal', 'max_new_tokens', {'hidden': True}]]
    registry = modules.MODULE_MAP['modules.HuggingFaceTransformers']['GenerateImageVideoText']['params']
    assert registry['enhancement_max_new_tokens']['hidden'] is True
    assert registry['enhancement_max_new_tokens']['default'] == 3584
    assert registry['max_new_tokens']['default'] == 256
    assert registry['min_new_tokens']['max'] == 2048


@pytest.mark.parametrize('index', [0, 1])
def test_private_plain_prompt_modes_remain_valid(index):
    import modules
    from modiff.ancillary_execution import validate_execution_purpose
    graph, _ = ancillary.preview_graph(modules.MODULE_MAP, index)
    before = deepcopy(graph)
    assert validate_execution_purpose(graph, modules.MODULE_MAP) == graph['runtimeHints']['executionPurpose']
    assert graph == before


@pytest.mark.parametrize('layout', [
    {'schemaVersion': 1, 'coordinateSpace': 'normalized_xyxy', 'regions': [
        {'id': 'fox', 'description': 'a red fox', 'targetBox': [0, 0, 0.5, 0.5]}]},
    {'schemaVersion': 1, 'coordinateSpace': 'normalized_xyxy', 'regions': []},
])
def test_private_preview_refuses_authored_layout_before_preparation(layout):
    import modules
    from modiff.ancillary_execution import validate_execution_purpose
    graph, _ = ancillary.preview_graph(modules.MODULE_MAP)
    graph['nodes']['proposal']['params']['layout_conditioning'] = {'value': deepcopy(layout)}
    before = deepcopy(graph)
    with pytest.raises(ValueError, match='plain prompt enhancement|authored layout'):
        validate_execution_purpose(graph, modules.MODULE_MAP)
    assert graph == before


@pytest.mark.parametrize('layout', [{}, [], False, 0, ''])
def test_private_preview_never_coerces_malformed_layout_to_none(layout):
    import modules
    from modiff.ancillary_execution import validate_execution_purpose
    graph, _ = ancillary.preview_graph(modules.MODULE_MAP)
    graph['nodes']['proposal']['params']['layout_conditioning'] = {'value': deepcopy(layout)}
    before = deepcopy(graph)
    with pytest.raises(ValueError, match='layout|JSON|object'):
        validate_execution_purpose(graph, modules.MODULE_MAP)
    assert graph == before


@pytest.mark.parametrize('fault', ['none', 'stale-declaration', 'foreign-parent-wire', 'foreign-child-wire',
    'wrong-media-output', 'missing-media', 'preview-neural-owner', 'swapped-child-role', 'missing-media-boundary'])
def test_actual_i2i_factory_stages_bind_only_declared_media_branch(fault):
    import modules
    from modiff.ancillary_execution import native_parent_association, validate_execution_purpose
    witness = json.loads((Path(__file__).parent / 'fixtures/flux3-i2i-split-stages-client.v1.json').read_text())
    parent = ancillary.parent_for(modules.MODULE_MAP, 1)
    declaration = parent['ancillaryOperations'][0]
    assert witness['binding']['parent'] == {**declaration['parent'], 'authoringContentHash': parent['contentHash']}
    assert witness['binding']['contentHash'] == declaration['contentHash']
    image, preview = deepcopy(witness['apiGraph']), deepcopy(witness['previewApiGraph'])
    binding = witness['binding'];owners, children = binding['parentRoleNodes'], binding['roleNodes']
    media = owners[declaration['bindings']['owner']['referenceSource']['role']]
    assert set(image['nodes']) == set(owners.values())
    assert set(preview['nodes']) == set(children.values()) | {media}
    assert validate_execution_purpose(image, modules.MODULE_MAP)['kind'] == 'workflow'
    assert validate_execution_purpose(preview, modules.MODULE_MAP)['kind'] == 'explicit_preview'
    target = image if fault in {'foreign-parent-wire', 'foreign-child-wire', 'missing-media-boundary'} else preview
    selected = target['runtimeHints']['executionPurpose']['bindings'][0]
    snapshot = target['runtimeHints']['workflowSnapshot']
    if fault == 'stale-declaration':
        selected['contentHash'] = 'studio-ancillary-v1-00000000'
    elif fault in {'foreign-parent-wire', 'foreign-child-wire'}:
        snapshot['nodes'].append({'id': 'foreign-owner', 'type': 'custom', 'position': {'x': 0, 'y': 0},
            'data': {'module': 'modules.HuggingFaceTransformers', 'action': 'LoadImageTextToTextModel'}})
        if fault == 'foreign-parent-wire':
            edge = {'source': media, 'sourceHandle': 'image', 'target': 'foreign-owner', 'targetHandle': 'unknown'}
        else:
            edge = {'source': 'foreign-owner', 'sourceHandle': 'model', 'target': children['proposal'], 'targetHandle': 'model'}
        snapshot['edges'].append({'id': 'foreign-incident-wire', **edge})
    elif fault == 'wrong-media-output':
        target['nodes'][children['proposal']]['params']['images']['sourceKey'] = 'mask'
    elif fault == 'missing-media':
        target['nodes'].pop(media)
    elif fault == 'preview-neural-owner':
        target['nodes'][owners['models']] = deepcopy(image['nodes'][owners['models']])
    elif fault == 'swapped-child-role':
        selected['roleNodes']['provider'], selected['roleNodes']['proposal'] = selected['roleNodes']['proposal'], selected['roleNodes']['provider']
    elif fault == 'missing-media-boundary':
        kept = [edge for edge in snapshot['edges']
                if not (edge.get('source') == media and str(edge.get('targetHandle', '')).endswith(':images'))]
        assert len(kept) == len(snapshot['edges']) - 1
        snapshot['edges'] = kept
    before = deepcopy(target)
    result = native_parent_association(target, selected, modules.MODULE_MAP)
    if fault == 'none':
        assert result is not None
        image_selected = image['runtimeHints']['executionPurpose']['bindings'][0]
        assert native_parent_association(image, image_selected, modules.MODULE_MAP) == result
    else:
        assert result is None
    assert target == before
