"""Ancillary-purpose selection through existing entrypoints; no models."""
from copy import deepcopy
from pathlib import Path
import asyncio
import hashlib
import json
from types import SimpleNamespace
import pytest
from modiff.server import WebServer
from modiff import native_component_lora


class ReachedPreflight(RuntimeError):
    pass


def graph():
    return {'sid': None, 'nodes': {'plain': {'module': 'modules.Text', 'action': 'Text',
        'params': {'text': {'value': 'kept'}}}}, 'paths': [['plain']]}


def test_claimed_purpose_is_not_silently_dropped():
    app = object.__new__(WebServer)
    claimed = {'schemaVersion': 1, 'kind': 'workflow', 'bindings': [], 'target': None}
    with pytest.raises(ValueError, match='(?i)ancillary|purpose'):
        app._coerce_runtime_hints({'resourceMode': 'expert', 'executionPurpose': claimed})


def test_invalid_claim_refuses_before_component_preflight(monkeypatch):
    app = object.__new__(WebServer)
    value = graph()
    value['runtimeHints'] = {'executionPurpose': {'schemaVersion': 1, 'kind': 'foreign', 'bindings': [], 'target': None}}
    def reached(*args, **kwargs):
        raise ReachedPreflight('original invalid purpose reached component preflight')
    monkeypatch.setattr(native_component_lora, 'preflight_component_graph', reached)
    with pytest.raises(ValueError, match='(?i)ancillary|purpose'):
        app._execute_graph(value)


def test_unclaimed_graph_keeps_existing_preflight_path(monkeypatch):
    app = object.__new__(WebServer)
    def reached(*args, **kwargs):
        raise ReachedPreflight('control reached existing component preflight')
    monkeypatch.setattr(native_component_lora, 'preflight_component_graph', reached)
    with pytest.raises(ReachedPreflight, match='control'):
        app._execute_graph(graph())


def test_unclaimed_hints_preserve_existing_values():
    app = object.__new__(WebServer)
    hints = {'resourceMode': 'expert', 'runInputHash': 'API-opaque-id', 'workflowCanvasEpoch': 4}
    before = deepcopy(hints)
    assert app._coerce_runtime_hints(hints) == hints == before


def require_native_parent_apis():
    """These private parent fixtures require the actual reviewed source classes.

    Keep the installed-release lane honest: metadata/owner tests must not fail
    inside their producer before reaching the boundary they claim to exercise.
    No class, runtime activation or executable availability is manufactured.
    """
    import diffusers
    from importlib import import_module
    required = {
        'Flux3ModularPipeline': 'diffusers.modular_pipelines.flux3.modular_pipeline',
        'Flux3DistilledModularPipeline': 'diffusers.modular_pipelines.flux3.modular_pipeline',
        'Flux3Transformer2DModel': 'diffusers.models.transformers.transformer_flux3',
    }
    for name, module_name in required.items():
        value = getattr(diffusers, name, None)
        if value is None:
            pytest.skip('FLUX 3 source APIs are absent from the installed release; run the pinned source audit.')
        if (not isinstance(value, type) or value.__module__ != module_name
                or getattr(import_module(module_name), name) is not value):
            pytest.fail('FLUX 3 fixture requires the actual native class identity: ' + name)


@pytest.fixture
def modules():
    require_native_parent_apis()
    import modules as registry
    return registry.MODULE_MAP


def parent_for(modules, index=0):
    require_native_parent_apis()
    from modiff.ancillary_execution import PRIVATE_PARENTS, resolve_parent
    identity = list(PRIVATE_PARENTS)[index]
    return resolve_parent(modules, identity)


def preview_graph(modules, index=0):
    from modiff.ancillary_execution import adapter_binding
    parent = parent_for(modules, index)
    contract = parent['ancillaryOperations'][0]
    owner_ids = {role: 'image-' + role for role, _key, _x, _y in parent['roles']}
    child_ids = {'provider': 'provider', 'proposal': 'proposal'}
    def node(key, **values):
        module, action = key.rsplit('.', 1)
        return {'module': module, 'action': action, 'params': {name: {'value': value} for name, value in values.items()}}
    descriptor, _binding = adapter_binding()
    nodes = {'provider': node('modules.HuggingFaceTransformers.LoadImageTextToTextModel',
        model_id={'source': 'hub', 'value': descriptor['repository']}, revision=descriptor['revision'], dtype='float32', device='cpu'),
        'proposal': node('modules.HuggingFaceTransformers.GenerateImageVideoText', operation='enhance_prompt',
            prompt=' Original raw prompt ', layout_conditioning=None)}
    nodes['proposal']['params']['model'] = {'sourceId': 'provider', 'sourceKey': 'model'}
    path = ['provider']
    reference = contract['bindings']['owner']['referenceSource']
    if reference:
        node_id = owner_ids[reference['role']]
        nodes[node_id] = node('modules.Image.Load', file=['first.png', 'second.png'])
        nodes['proposal']['params']['images'] = {'sourceId': node_id, 'sourceKey': 'image'}
        path.append(node_id)
    path.append('proposal')
    binding = {'contractId': contract['id'], 'contentHash': contract['contentHash'],
        'parent': {**contract['parent'], 'authoringContentHash': parent['contentHash']},
        'parentRoleNodes': owner_ids, 'roleNodes': child_ids}
    scope = {'schemaVersion': 1, 'kind': 'explicit_preview', 'bindings': [binding],
        'target': {'nodeId': 'proposal', 'outputField': 'proposal_summary'}}
    return {'sid': None, 'nodes': nodes, 'paths': [path], 'runtimeHints': {'executionPurpose': scope}}, parent


@pytest.mark.parametrize('index', range(4))
def test_four_private_parents_have_exact_hash_covered_optional_declaration(modules, index):
    from modiff.ancillary_execution import validate_execution_purpose
    from modiff.studio_execution_specs import _hash_string, _stable_json
    value, parent = preview_graph(modules, index)
    unsigned = {key: item for key, item in parent.items() if key != 'contentHash'}
    assert parent['contentHash'] == 'studio-authoring-v1-' + _hash_string(_stable_json(unsigned))
    child = parent['ancillaryOperations'][0]
    assert child['contentHash'] == 'studio-ancillary-v1-' + _hash_string(_stable_json({k:v for k,v in child.items() if k != 'contentHash'}))
    assert 'authoringContentHash' not in child['parent']
    assert child['sourceRegions'] is False
    assert 'layoutPolicy' not in child and 'layoutRequestSuffixSha256' not in child['verification']
    assert child['verification']['supportedPromptModes'] == ['t2i', 'i2i']
    assert validate_execution_purpose(value, modules) == value['runtimeHints']['executionPurpose']


def test_valid_selection_survives_normalization_without_coercion(modules):
    value, _ = preview_graph(modules)
    app = object.__new__(WebServer)
    actual = app._coerce_runtime_hints(value['runtimeHints'])
    assert actual == value['runtimeHints']
    assert actual is not value['runtimeHints']
    assert actual['executionPurpose'] is not value['runtimeHints']['executionPurpose']


@pytest.mark.parametrize('key', ['authoringContentHash', 'baseSpecContentHash', 'executionProfileId', 'task'])
def test_mixed_or_stale_parent_refuses(modules, key):
    from modiff.ancillary_execution import validate_execution_purpose
    value, _ = preview_graph(modules)
    parent = value['runtimeHints']['executionPurpose']['bindings'][0]['parent']
    parent[key] = parent[key].replace('v1-', 'v1-a', 1) if 'Hash' in key else 'foreign'
    with pytest.raises(ValueError, match='ancillary'):
        validate_execution_purpose(value, modules)


def test_extra_image_owner_never_silently_pruned(modules):
    from modiff.ancillary_execution import validate_execution_purpose
    value, _ = preview_graph(modules)
    value['nodes']['image-models'] = {'module':'modules.ModularDiffusers','action':'ModelsLoader','params':{}}
    value['paths'].insert(0, ['image-models'])
    before = deepcopy(value)
    with pytest.raises(ValueError, match='image owner|unrelated'):
        validate_execution_purpose(value, modules)
    assert value == before


def test_real_reference_order_and_no_owner_access_wire(modules):
    from modiff.ancillary_execution import validate_execution_purpose
    value, _ = preview_graph(modules, 1)
    before = deepcopy(value)
    validate_execution_purpose(value, modules)
    assert value == before
    assert value['nodes']['image-loadImage']['params']['file']['value'] == ['first.png','second.png']
    value['nodes']['proposal']['params']['images'] = {'sourceId':'provider','sourceKey':'receipt'}
    with pytest.raises(ValueError, match='reference'):
        validate_execution_purpose(value, modules)


def test_bad_claim_refuses_auto_before_probe(modules):
    app = object.__new__(WebServer)
    app.modules = modules
    app._runtime_fingerprint = lambda: pytest.fail('must not probe before purpose validation')
    value = graph()
    value['runtimeHints'] = {'executionPurpose': None}
    with pytest.raises(ValueError, match='ancillary'):
        app._build_workflow_auto_plan(value)


def workflow_graph(modules, index=0):
    from modiff.studio_execution_specs import (
        STUDIO_EXECUTION_SPEC_DEFINITIONS, _public_spec, _execution_spec_role_params)
    preview, parent = preview_graph(modules, index)
    binding = preview['runtimeHints']['executionPurpose']['bindings'][0]
    ids = binding['parentRoleNodes']
    spec = _public_spec(parent['baseSpecId'], STUDIO_EXECUTION_SPEC_DEFINITIONS[parent['baseSpecId']])
    nodes = {}
    for role, key, _x, _y in parent['roles']:
        module, action = key.rsplit('.', 1)
        params = _execution_spec_role_params(spec, key, modules[module][action])
        nodes[ids[role]] = {'module': module, 'action': action,
            'params': {name: {'value': deepcopy(field.get('default'))}
                       for name, field in params.items() if field.get('display') != 'output'}}
    for source, output, target, input_ in parent['edges']:
        nodes[ids[target]]['params'][input_] = {'sourceId': ids[source], 'sourceKey': output}
    for role, name, source in parent['bindings']:
        if source == 'pipelineClass':
            nodes[ids[role]]['params'][name] = {'value': spec['pipelineClass']}
    # Topological fixture ordering follows the actual declared edges, rather
    # than diagram positions. This never executes a supplier.
    order = []
    remaining = list(nodes)
    while remaining:
        ready = [node_id for node_id in remaining if all(
            item.get('sourceId') is None or item['sourceId'] in order
            for item in nodes[node_id]['params'].values())]
        assert ready, 'declared parent fixture is cyclic'
        order.extend(ready)
        remaining = [node_id for node_id in remaining if node_id not in ready]
    scope = {'schemaVersion': 1, 'kind': 'workflow', 'bindings': [binding], 'target': None}
    hints = {'executionPurpose': scope, 'modelType': spec['modelType'], 'mode': spec['mode'],
        'studioExecutionSpec': {'schemaVersion': 1, 'id': spec['id'],
                              'contentHash': spec['contentHash'], 'nodes': ids}}
    return {'sid': None, 'nodes': nodes, 'paths': [order], 'runtimeHints': hints}


@pytest.mark.parametrize('index', range(4))
def test_workflow_preserves_real_main_spec_and_unrelated_branch(modules, index):
    from modiff.ancillary_execution import validate_execution_purpose
    from modiff.studio_execution_specs import assert_studio_execution_graph
    value = workflow_graph(modules, index)
    value['nodes'].update(graph()['nodes'])
    value['paths'].append(['plain'])
    before = deepcopy(value)
    validate_execution_purpose(value, modules)
    assert_studio_execution_graph(value, value['runtimeHints'])
    assert value == before
    value['runtimeHints']['studioExecutionSpec']['contentHash'] = 'studio-spec-v1-00000000'
    # The existing late managed-graph receipt check remains independently live.
    with pytest.raises(RuntimeError, match='specification'):
        assert_studio_execution_graph(value, value['runtimeHints'])


def test_workflow_with_preview_sink_refuses_instead_of_pruning(modules):
    from modiff.ancillary_execution import validate_execution_purpose
    value = workflow_graph(modules)
    extra, _parent = preview_graph(modules)
    value['nodes'].update(extra['nodes'])
    value['paths'] += extra['paths']
    before = deepcopy(value)
    with pytest.raises(ValueError, match='Preview-only'):
        validate_execution_purpose(value, modules)
    assert value == before


@pytest.mark.parametrize('role', ['models', 'guider'])
def test_workflow_claim_binds_actual_selected_pipeline_literals(modules, role):
    from modiff.ancillary_execution import validate_execution_purpose
    value = workflow_graph(modules, 2)
    node_id = value['runtimeHints']['executionPurpose']['bindings'][0]['parentRoleNodes'][role]
    value['nodes'][node_id]['params']['model_type'] = {'value': 'StableDiffusionXLModularPipeline'}
    with pytest.raises(ValueError, match='pipeline selection'):
        validate_execution_purpose(value, modules)


@pytest.mark.parametrize('field', ['proposal_summary', 'result'])
def test_static_output_cannot_be_supplied_as_input(modules, field):
    from modiff.ancillary_execution import validate_execution_purpose
    value, _parent = preview_graph(modules)
    value['nodes']['proposal']['params'][field] = {'value': 'forged'}
    with pytest.raises(ValueError, match='output'):
        validate_execution_purpose(value, modules)


@pytest.mark.parametrize('mutation', [
    'foreign_child_hash', 'unknown_parent', 'child_roles', 'parent_roles',
    'wrong_output', 'wrong_edge', 'mutable_revision', 'local_provider',
    'extra_param', 'video', 'inline_images', 'source_region', 'cycle',
    'wrong_order', 'missing_path', 'loops'])
def test_preview_related_wrong_scopes_refuse_without_mutation(modules, mutation):
    from modiff.ancillary_execution import validate_execution_purpose
    value, _parent = preview_graph(modules)
    scope = value['runtimeHints']['executionPurpose']
    binding = scope['bindings'][0]
    proposal = value['nodes']['proposal']['params']
    if mutation == 'foreign_child_hash': binding['contentHash'] = 'studio-ancillary-v1-00000000'
    elif mutation == 'unknown_parent': binding['parent']['authoringRecipeId'] = 'unknown'
    elif mutation == 'child_roles': binding['roleNodes']['unexpected'] = 'unused'
    elif mutation == 'parent_roles': binding['parentRoleNodes'].pop('prompt')
    elif mutation == 'wrong_output': scope['target']['outputField'] = 'text'
    elif mutation == 'wrong_edge': proposal['model']['sourceKey'] = 'receipt'
    elif mutation == 'mutable_revision': value['nodes']['provider']['params']['revision']['value'] = 'main'
    elif mutation == 'local_provider': value['nodes']['provider']['params']['model_id']['value']['source'] = 'local'
    elif mutation == 'extra_param': proposal['not_declared'] = {'value': 1}
    elif mutation == 'video': proposal['video'] = {'value': 'foreign.mp4'}
    elif mutation == 'inline_images': proposal['images'] = {'value': ['foreign.png']}
    elif mutation == 'source_region':
        proposal['layout_conditioning']['value'] = {'schemaVersion': 1,
            'coordinateSpace': 'normalized_xyxy',
            'regions': [{'id': 'region1', 'description': 'subject', 'targetBox': [0, 0, 0.5, 0.5],
                         'source': {'referenceIndex': 0, 'box': [0, 0, 0.5, 0.5]}}]}
    elif mutation == 'cycle': value['nodes']['provider']['params']['model_id'] = {'sourceId': 'proposal', 'sourceKey': 'text'}
    elif mutation == 'wrong_order': value['paths'][0].reverse()
    elif mutation == 'missing_path': value['paths'] = [['proposal']]
    elif mutation == 'loops': value['loops'] = {'unexpected': {}}
    before = deepcopy(value)
    with pytest.raises(ValueError, match='ancillary'):
        validate_execution_purpose(value, modules)
    assert value == before


@pytest.mark.parametrize('mutation', ['null', 'boolean_version', 'unknown_key', 'duplicate',
                                   'role_collision', 'huge', 'nonfinite', 'non_json'])
def test_selection_closed_json_bounds(modules, mutation):
    from modiff.ancillary_execution import parse_execution_purpose
    value, _parent = preview_graph(modules)
    scope = value['runtimeHints']['executionPurpose']
    if mutation == 'null': scope = None
    elif mutation == 'boolean_version': scope['schemaVersion'] = True
    elif mutation == 'unknown_key': scope['release'] = True
    elif mutation == 'duplicate': scope['kind'] = 'workflow'; scope['target'] = None; scope['bindings'] *= 2
    elif mutation == 'role_collision': scope['bindings'][0]['roleNodes']['proposal'] = 'provider'
    elif mutation == 'huge': scope['target']['nodeId'] = 'x' * 131073
    elif mutation == 'nonfinite': scope['schemaVersion'] = float('nan')
    elif mutation == 'non_json': scope['schemaVersion'] = object()
    with pytest.raises(ValueError, match='ancillary'):
        parse_execution_purpose(scope)


def test_invalid_claim_cannot_enter_queue_or_persistence(modules):
    app = object.__new__(WebServer)
    app.modules = modules
    app._runtime_mutation_gate = None
    app.queued_tasks = {}
    app.task_graphs = {}
    app._persist_supervisor_queue_state = lambda **_kwargs: pytest.fail('must not persist')
    value, _parent = preview_graph(modules)
    value['runtimeHints']['executionPurpose']['bindings'][0]['contentHash'] = 'studio-ancillary-v1-00000000'
    with pytest.raises(ValueError, match='ancillary'):
        asyncio.run(app.queue_task(lambda _graph: None, (value,), None, None, name='Graph execution'))
    assert app.queued_tasks == app.task_graphs == {}


def test_invalid_auto_request_refuses_before_busy_or_lease(modules):
    app = object.__new__(WebServer)
    app.modules = modules
    app._workflow_auto_inspection_busy = lambda **_kwargs: pytest.fail('must not inspect busy state')
    value, _parent = preview_graph(modules)
    value['runtimeHints']['executionPurpose']['bindings'][0]['contentHash'] = 'studio-ancillary-v1-00000000'
    async def body(): return {'schemaVersion': 1, 'graph': value}
    response = asyncio.run(app.workflow_auto_resource_plan(SimpleNamespace(json=body)))
    assert response.status == 400
    assert 'ancillary' in json.loads(response.body)['message']


def test_authored_target_layout_preserves_exact_dto_and_refuses_rounding_before_preparation(modules):
    from modiff.ancillary_execution import validate_execution_purpose
    value, _parent = preview_graph(modules, 4)
    value['nodes']['proposal']['params']['layout_conditioning']['value'] = {
        'schemaVersion': 2, 'coordinateSpace': 'normalized_xyxy', 'regions': [
            {'id': 'subject', 'description': ' paper crane ', 'action': 'place',
             'targetBox': [0, 0.125, 0.5, 0.625]}]}
    before = deepcopy(value)
    assert validate_execution_purpose(value, modules) == value['runtimeHints']['executionPurpose']
    assert value == before
    value['nodes']['proposal']['params']['layout_conditioning']['value']['regions'][0]['targetBox'][0] = 0.0001
    before = deepcopy(value)
    with pytest.raises(ValueError, match='never rounded'):
        validate_execution_purpose(value, modules)
    assert value == before


def test_authored_source_preview_keeps_binding_and_rejects_missing_actual_media_index(modules):
    from modiff.ancillary_execution import validate_execution_purpose
    value, _parent = preview_graph(modules, 5)
    layout = {'schemaVersion': 2, 'coordinateSpace': 'normalized_xyxy', 'regions': [{
        'id': 'subject', 'description': 'A subject.', 'action': 'remove', 'targetBox': None,
        'source': {'referenceIndex': 1, 'box': [0, 0, 1, 1], 'expectedReferenceId': 'sha256:' + 'a' * 64}}]}
    value['nodes']['proposal']['params']['layout_conditioning']['value'] = layout
    before = deepcopy(value)
    assert validate_execution_purpose(value, modules) == value['runtimeHints']['executionPurpose']
    assert value == before
    layout['regions'][0]['source']['referenceIndex'] = 2
    with pytest.raises(ValueError, match='ordered media context'):
        validate_execution_purpose(value, modules)


def test_legacy_plain_parent_dto_and_both_hashes_stay_exact(modules):
    fixture = json.loads((Path(__file__).parent / 'fixtures/prompt-enhancement-legacy-parent.v1.json').read_text(encoding='utf-8'))
    assert json.loads(json.dumps(parent_for(modules, 1))) == fixture['authoring']
    assert parent_for(modules, 0)['contentHash'] == 'studio-authoring-v1-c39b36d8'
    assert parent_for(modules, 1)['contentHash'] == 'studio-authoring-v1-d43668d0'
    assert parent_for(modules, 1)['ancillaryOperations'][0]['contentHash'] == 'studio-ancillary-v1-95fc4619'


def test_explicit_fp32_t2i_parent_is_available_without_rewriting_the_legacy_recipe(modules):
    from modiff.ancillary_execution import resolve_parent
    parent = resolve_parent(modules, 'flux3-klein:text-to-image:prompt-preview-fp32-authoring:v1')
    child = parent['ancillaryOperations'][0]
    values = {(role, field): value for role, field, value in child['initialValues']}
    assert values['provider', 'dtype'] == 'float32'
    assert values['provider', 'device'] == 'cuda:0'
    assert values['provider', 'quantization_mode'] == 'none'
    assert values['proposal', 'max_input_tokens'] == 4096
    assert child['bindings']['owner']['referenceSource'] is None
    assert child['verification']['supportedPromptModes'] == ['t2i', 'i2i']


def test_explicit_fp32_preview_dispatch_rejects_changed_fixed_inputs(modules):
    from modiff.ancillary_execution import PRIVATE_PARENTS, validate_execution_purpose
    index = list(PRIVATE_PARENTS).index('flux3-klein:text-to-image:prompt-preview-fp32-authoring:v1')
    value, parent = preview_graph(modules, index)
    declaration = parent['ancillaryOperations'][0]
    roles = value['runtimeHints']['executionPurpose']['bindings'][0]['roleNodes']
    for role, field, selected in declaration['initialValues']:
        value['nodes'][roles[role]]['params'][field] = {'value': deepcopy(selected)}
    assert validate_execution_purpose(value, modules) == value['runtimeHints']['executionPurpose']
    for role, field, wrong in [('provider', 'dtype', 'bfloat16'),
                                ('provider', 'quantization_mode', 'bnb_4bit'),
                                ('provider', 'device', 'cpu'),
                                ('proposal', 'max_input_tokens', 4095),
                                ('proposal', 'max_input_tokens', 4096.0)]:
        changed = deepcopy(value)
        changed['nodes'][roles[role]]['params'][field] = {'value': wrong}
        before = deepcopy(changed)
        with pytest.raises(ValueError, match='FP32 Preview input selection changed'):
            validate_execution_purpose(changed, modules)
        assert changed == before


def test_fp32_variant_preserves_all_eight_prior_full_dtos(modules):
    from modiff.ancillary_execution import resolve_parent
    # Captured from the reviewed structured-system source before adding FP32 T2I.
    expected = {
        'flux3-klein:text-to-image:prompt-preview-authoring:v1': '32aeb04804192951ed52461ef8b6621971f0f1776b39f322eeeadd5977433f94',
        'flux3-klein:edit-image:prompt-preview-authoring:v1': 'dd9307b5470dac4d1455edf9c98df788f1b2ced8454e4c38d98de0d4ccb5697c',
        'flux3-klein-base:text-to-image:prompt-preview-authoring:v1': '29b9708b61b0115653417e7be2869b0073981bce6f67642cd95ea5342a09334c',
        'flux3-klein-base:edit-image:prompt-preview-authoring:v1': '6e8299402d516df85f342a3ce80b723f22b73dc130f1666e6e72c75176c1d96b',
        'flux3-klein:text-to-image:prompt-preview-authoring:v2': '40483e6dc13a6b9aaf07584b6e8da06e1adab50fe765b110bfd3ec8dc344df12',
        'flux3-klein:edit-image:prompt-preview-authoring:v2': '8fae8fa3fbdc874b578356728a2a67451802a3775f67ba82a0c5938231b6ea88',
        'flux3-klein-base:text-to-image:prompt-preview-authoring:v2': 'f783677c6f19ae45054bd7c19052b387899056851246b21ac91e95e7719b57c0',
        'flux3-klein-base:edit-image:prompt-preview-authoring:v2': '1bb43007634b0a74f0dbe44868859abe346a6806e6c3364060d2c3ee18c0e665',
    }
    for identity, digest in expected.items():
        dto = resolve_parent(modules, identity)
        assert hashlib.sha256(json.dumps(dto, sort_keys=True, ensure_ascii=False,
            separators=(',', ':')).encode('utf-8')).hexdigest() == digest


@pytest.mark.parametrize('index', range(4))
def test_versioned_preservation_is_distinct_and_legacy_preview_still_refuses_layout(modules, index):
    from modiff.ancillary_execution import validate_execution_purpose
    from modiff import prompt_enhancement as adapter
    legacy, structured = parent_for(modules, index), parent_for(modules, index + 4)
    for key in ('baseSpecId', 'baseSpecContentHash', 'executionProfileIds', 'roles', 'edges', 'bindings', 'initialValues'):
        assert legacy[key] == structured[key]
    assert structured['id'] == legacy['id'].replace(':v1', ':v2')
    assert structured['contentHash'] != legacy['contentHash']
    child = structured['ancillaryOperations'][0]
    assert child['layoutPolicy'] == 'preserve-authored-v1'
    assert child['verification']['supportedPromptModes'] == ['t2i', 'i2i', 't2i_bbox', 'i2i_bbox']
    assert child['verification']['layoutRequestSuffixSha256'].startswith('sha256:')
    _raw, prompts = adapter._json_file(adapter.SYSTEM_PROMPTS, 32 * 1024)
    assert legacy['ancillaryOperations'][0]['verification']['systemPromptSha256ByMode'] == {
        mode: hashlib.sha256(text.encode('utf-8')).hexdigest() for mode, text in prompts.items()}
    assert child['verification']['systemPromptSha256ByMode'] == {
        mode: hashlib.sha256(adapter.request_system_prompt(prompts, mode).encode('utf-8')).hexdigest()
        for mode in prompts}
    value, _ = preview_graph(modules, index)
    value['nodes']['proposal']['params']['layout_conditioning']['value'] = {
        'schemaVersion': 2, 'coordinateSpace': 'normalized_xyxy', 'regions': []}
    with pytest.raises(ValueError, match='plain prompt enhancement'):
        validate_execution_purpose(value, modules)


@pytest.mark.parametrize('field,value', [('modelWeight', {}), ('transformersVersion', []),
                                      ('processorClass', False), ('chatTemplateSha256', 'not-a-hash')])
def test_malformed_selected_declaration_refuses_before_scope_use(modules, monkeypatch, field, value):
    from modiff import prompt_enhancement, ancillary_execution
    original = prompt_enhancement._json_file
    def malformed(path, maximum):
        raw, result = original(path, maximum)
        if path == prompt_enhancement.DESCRIPTOR:
            result[field] = value
        return raw, result
    monkeypatch.setattr(prompt_enhancement, '_json_file', malformed)
    with pytest.raises(ValueError, match='ancillary'):
        ancillary_execution.adapter_binding()


def test_parent_guider_roles_match_existing_closed_vocabulary(modules):
    for index in (2, 3):
        parent = parent_for(modules, index)
        assert ('guider', 'modules.ModularDiffusers.Guider', -360, 160) in parent['roles']
        assert ['guider', 'guider_out', 'denoise', 'guider'] in [list(row) for row in parent['edges']]
        assert all(row[0] != 'guidance' for row in parent['roles'])


@pytest.mark.parametrize('index', range(4))
def test_selected_t2i_nf4_recipe_is_visible_and_keeps_registered_generation_default(modules, index):
    from modules.HuggingFaceTransformers import main as nodes
    parent = parent_for(modules, index)
    declaration = parent['ancillaryOperations'][0]
    values = {(role, field): value for role, field, value in declaration['initialValues']}
    runtime = declaration['verification']['runtimeInputs']
    request = declaration['verification']['requestInputs']
    recipe = {(runtime['role'], runtime['dtypeField']): 'bfloat16',
              (runtime['role'], runtime['deviceField']): 'cuda:0',
              (runtime['role'], runtime['quantizationField']): 'bnb_4bit',
              (request['role'], request['maxInputTokensField']): 4096}
    if declaration['parent']['task'] == 'text_to_image':
        assert {key: values[key] for key in recipe} == recipe
    else:
        assert not set(values).intersection(recipe)
    assert (request['role'], request['maxNewTokensField']) not in values
    assert nodes.GenerateImageVideoText.params[request['maxNewTokensField']]['default'] == 3584
    assert nodes.GenerateImageVideoText.params[request['maxNewTokensField']]['max'] == 3584
    assert nodes.LoadImageTextToTextModel.params['dtype']['default'] == 'float32'
    assert nodes.LoadImageTextToTextModel.params['device']['default'].startswith('cpu')
    assert nodes.LoadImageTextToTextModel.params['quantization_mode']['default'] == 'none'


def test_private_parent_fixture_matches_actual_api_availability_without_activation(monkeypatch):
    """Installed refusal and reviewed-source metadata are separate live controls."""
    import diffusers
    import transformers
    import modules as registry
    from modiff import optimization_packages
    from modiff.ancillary_execution import PRIVATE_PARENTS, resolve_parent
    def factory_forbidden(*_args, **_kwargs):
        pytest.fail('Private parent metadata must not activate or construct a model')
    monkeypatch.setattr(optimization_packages, 'activate_runtime_overlay', factory_forbidden)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained', factory_forbidden)
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', factory_forbidden)
    monkeypatch.setattr(diffusers.DiffusionPipeline, 'from_pretrained', factory_forbidden)
    identity = next(iter(PRIVATE_PARENTS))
    if getattr(diffusers, 'Flux3DistilledModularPipeline', None) is None:
        # The real production resolver stays closed on the ordinary installed
        # wheel. The shared test producer must classify its own missing API.
        with pytest.raises(ValueError, match='Unknown Diffusers modular pipeline class'):
            resolve_parent(registry.MODULE_MAP, identity)
        with pytest.raises(pytest.skip.Exception, match='source APIs are absent'):
            parent_for(registry.MODULE_MAP)
    else:
        parent = parent_for(registry.MODULE_MAP)
        assert parent == resolve_parent(registry.MODULE_MAP, identity)
        assert parent['ancillaryOperations'] and parent['ancillaryOperations'][0]['parent']['pipelineClass'] == 'Flux3DistilledModularPipeline'


@pytest.mark.parametrize('name', ['Flux3ModularPipeline', 'Flux3DistilledModularPipeline', 'Flux3Transformer2DModel'])
def test_private_parent_fixture_refuses_foreign_export_identity(modules, monkeypatch, name):
    import diffusers
    monkeypatch.setattr(diffusers, name, object())
    with pytest.raises(pytest.fail.Exception, match='actual native class identity'):
        parent_for(modules)
