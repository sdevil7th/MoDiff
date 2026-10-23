"""Native producer transport: actual tiny CPU dispatch, never a live task claim."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aiohttp.test_utils import make_mocked_request

import test_prompt_enhancement as stage_one
import test_prompt_enhancement_ownership as owner_tests
from modiff import prompt_enhancement as adapter
from modiff import prompt_enhancement_summary as transport
from modiff.backend_source_identity import process_backend_source_identity
from modiff.native_component_lora import native_component_graph_scope
from modiff.NodeBase import memory_manager
from modiff.workflow_auto_lifecycle import release_owner_caches
from modules.HuggingFaceTransformers import main as nodes

actual_pup = stage_one.actual_pup
isolated_manager = owner_tests.isolated_manager


def bound_pair(monkeypatch, *, hints=None):
    def dispatch(app, graph, node_id):
        app.instance = 'explicit-pup-unit-worker-' + str(os.getpid())
        app.backend_source_identity = process_backend_source_identity()
        app.current_task['runtimeHints'] = deepcopy(hints or {})
        app.current_task['current_node'] = node_id
        if node_id == 'pup-proposal' and not hasattr(app, '_unit_original_expectation'):
            from modiff import prompt_enhancement as adapter
            from modiff.workflow_auto_resource import workflow_graph_hash
            handle = app.node_cache['pup-owner'].output['model']
            config = handle['model'].config
            declaration = adapter.DESCRIPTOR.read_bytes()
            params = graph['nodes'][node_id]['params']
            app._unit_original_expectation = {
                'proofScope': 'Independently captured before Generate dispatch; explicit synthetic local tiny checkpoint',
                'producer': {'taskId': app.current_task['task_id'], 'attemptIndex': app.current_task['attempt_index'],
                    'sid': graph['sid'], 'node': {'id': node_id, **{key: graph['nodes'][node_id][key] for key in ('module', 'action')}},
                    'workerInstance': app.instance, 'backendSourceFingerprint': app.backend_source_identity['fingerprint'],
                    'graphHash': workflow_graph_hash(graph), 'runtimeHints': deepcopy(app.current_task['runtimeHints'])},
                'graph': deepcopy(graph), 'declarationBytesUtf8': declaration.decode('utf-8'),
                'declaration': json.loads(declaration), 'declarationSha256': 'sha256:' + hashlib.sha256(declaration).hexdigest(),
                'modelReceiptFromOwnerBeforeGenerate': deepcopy(handle['receipt']),
                'modelConfigFromActualPreCallHandle': {'modelType': config.model_type, 'architectures': deepcopy(config.architectures)},
                'rawAuthoredPrompt': params['prompt']['value'], 'rawAuthoredLayout': None,
                'actualControlExpectation': {'max_new_tokens': 2, 'min_new_tokens': 2, 'do_sample': False,
                    'num_beams': 1, 'repetition_penalty': 1.0, 'return_dict_in_generate': False, 'use_chat_template': True},
                'orderedSourceReferenceExpectation': [],
            }
        with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']):
            app.execute_node(node_id, graph['nodes'][node_id], None, quiet=True)
    monkeypatch.setattr(owner_tests, 'dispatch', dispatch)
    return owner_tests.cached_pair(monkeypatch)


def cache_response(app, field):
    request = make_mocked_request('GET', '/cache/pup-proposal/' + field,
                                  match_info={'node': 'pup-proposal', 'field': field})
    return asyncio.run(app.cache(request))


def persist(label, app, summary):
    output = os.environ.get('MODIFF_PUP_SUMMARY_FIXTURES')
    if not output:
        return
    from modiff import prompt_enhancement as adapter
    raw = adapter.DESCRIPTOR.read_bytes()
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    with (path / (label + '.json')).open('x', encoding='utf-8') as stream:
        json.dump({'proofScope': 'Actual normal WebServer/NodeBase unit task with pinned processor and synthetic tiny local weights; explicit unit worker, no HTTP or trained model claim',
            'summary': summary, 'expectedDeclaration': json.loads(raw),
            'expectedDeclarationSha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
            'unitCurrentTaskId': app.current_task['task_id'],
            'externalPreCallExpectations': deepcopy(app._unit_original_expectation)}, stream, indent=2)


def test_bound_cold_producer_requires_closed_summary(actual_pup, monkeypatch):
    app, _ = bound_pair(monkeypatch)
    output = app.node_cache['pup-proposal'].output
    summary = output.get('proposal_summary')
    assert type(summary) is str and summary
    value = json.loads(summary)
    assert value['producer']['taskId'] == 'pup-owner-unit'
    assert value['producer']['sid'] is None and value['producer']['recipe'] is None
    assert all(value is None for value in value['producer']['uiContext'].values())
    assert value['proposal']['text'] == output['text'] == output['result']['text']
    persist('cold-no-ui', app, summary)


def test_warm_new_task_keeps_original_producer_bytes(actual_pup, monkeypatch):
    _, calls = actual_pup
    hints = {'clientRunId': 'unit-A', 'runInputHash': 'opaque API correlation', 'recipe': {'forged': True}}
    app, graph = bound_pair(monkeypatch, hints=hints)
    summary = app.node_cache['pup-proposal'].output.get('proposal_summary')
    count = len(calls)
    app.current_task.update(task_id='explicit-unit-task-B', attempt_index=1)
    hints.update(clientRunId='unit-B', runInputHash='distinct API task B')
    owner_tests.dispatch(app, graph, 'pup-owner')
    owner_tests.dispatch(app, graph, 'pup-proposal')
    current = app.node_cache['pup-proposal'].output.get('proposal_summary')
    assert len(calls) == count and app.node_cache['pup-proposal']._has_changed is False
    assert type(current) is str and current == summary
    producer = json.loads(current)['producer']
    assert producer['taskId'] == 'pup-owner-unit' and producer['uiContext']['clientRunId'] == 'unit-A'
    assert producer['recipe'] is None and app.current_task['runtimeHints']['clientRunId'] == 'unit-B'
    persist('warm-task-B-original-A', app, current)


def test_actual_cache_before_and_after_detached_retention_requires_static_identity(actual_pup, monkeypatch):
    app, _ = bound_pair(monkeypatch)
    first = cache_response(app, 'proposal_summary')
    outputs = app.node_cache['pup-proposal'].output
    selected = [key for key in ('text', 'result', 'proposal_summary') if key in outputs]
    del outputs
    release_owner_caches(app, {'ownerIds': ['pup-owner'],
        'nodeIds': app._dependent_cache_nodes(['pup-owner']),
        'retainOutputs': {'pup-proposal': selected}}, memory_manager)
    second = cache_response(app, 'proposal_summary')
    assert (first.status, second.status) == (200, 200)
    assert first.body == second.body
    assert cache_response(app, 'result').status == 400
    persist('retained-summary', app, second.body.decode('utf-8'))
    holder = app.node_cache['pup-proposal']
    from modiff.workflow_auto_lifecycle import _RetainedPromptProposal
    assert type(holder) is _RetainedPromptProposal
    owner_tests.dispatch(app, app.current_task['args'][0], 'pup-owner')
    owner_tests.dispatch(app, app.current_task['args'][0], 'pup-proposal')
    assert type(app.node_cache['pup-proposal']) is nodes.GenerateImageVideoText
    assert app.node_cache['pup-proposal'] is not holder and holder.output['proposal_summary'] == first.body.decode()




def context(monkeypatch, *, prompt='  A red cube.  ', layout=None):
    node = nodes.GenerateImageVideoText('bound-unit-proposal')
    graph, task, app = stage_one.owned_generation(monkeypatch, node)
    return node, graph, task, app, prompt, layout


def test_standard_and_direct_enhancement_keep_values_without_producer(actual_pup):
    from PIL import Image
    handle, _ = actual_pup
    for operation in ['standard', 'enhance_prompt']:
        node = nodes.GenerateImageVideoText('direct-' + operation)
        values = dict(model=handle, prompt='A red cube.', operation=operation,
            max_new_tokens=2, enhancement_max_new_tokens=2, min_new_tokens=2)
        if operation == 'standard':
            values['images'] = [Image.new('RGB', (256, 256), 'red')]
        node(**values)
        assert set(node.output) == {'text', 'result', 'proposal_summary', 'proposal_diagnostic'}
        assert node.output['proposal_summary'] is None
        assert node.output['proposal_diagnostic'] is None
        assert node.output['text'] == node.output['result']['text']
        direct = node.execute(**values)
        assert direct == node.output and direct['proposal_summary'] is None
        assert node._skip_params_check is False


@pytest.mark.parametrize('field,value', [('attempt_index', None), ('attempt_index', True),
    ('attempt_index', -1), ('attempt_index', 1001), ('task_id', 'x' * 129)])
def test_bound_producer_rejects_invalid_task_observations(monkeypatch, field, value):
    node, graph, task, app, prompt, layout = context(monkeypatch)
    task[field] = value
    with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']):
        observer = adapter._OwnedGeneration(node)
        assert observer.task is task
        with pytest.raises(ValueError):
            transport.BoundSummary(observer, b'{}', {}, prompt, layout)


@pytest.mark.parametrize('hints', [[], {'clientRunId': ''}, {'runInputHash': 7},
    {'workflowTabId': 'x' * 513}, {'workflowCanvasEpoch': True}, {'workflowFormEpoch': -1},
    {'workflowCanvasEpoch': 2**53}])
def test_bound_ui_correlation_is_strict_and_never_coerced(monkeypatch, hints):
    node, graph, task, app, prompt, layout = context(monkeypatch)
    task['runtimeHints'] = hints
    with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']), pytest.raises(ValueError):
        transport.BoundSummary(adapter._OwnedGeneration(node), b'{}', {}, prompt, layout)


@pytest.mark.parametrize('mismatch', ['task_graph', 'current_node', 'cached_node', 'source_action', 'sid', 'no_scope'])
def test_unbound_or_foreign_context_cannot_mint_a_producer(monkeypatch, mismatch):
    node, graph, task, app, _prompt, _layout = context(monkeypatch)
    if mismatch == 'task_graph':
        task['args'] = [deepcopy(graph)]
    elif mismatch == 'current_node':
        task['current_node'] = 'different-node'
    elif mismatch == 'cached_node':
        app.node_cache[node.node_id] = object()
    elif mismatch == 'source_action':
        graph['nodes'][node.node_id]['action'] = 'Other'
    elif mismatch == 'sid':
        task['sid'] = 'foreign'
    if mismatch == 'no_scope':
        assert adapter._OwnedGeneration(node).task is None
    else:
        with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']):
            assert adapter._OwnedGeneration(node).task is None


@pytest.mark.parametrize('change', ['graph', 'worker', 'attempt', 'hints', 'declaration', 'stop'])
def test_final_drift_or_stop_cannot_publish_a_new_summary(actual_pup, monkeypatch, change):
    handle, _ = actual_pup
    node, graph, task, app, prompt, _layout = context(monkeypatch)
    original = handle['model'].generate
    calls = []
    def mutated(**kwargs):
        calls.append(True)
        value = original(**kwargs)
        if change == 'graph':
            graph['nodes'][node.node_id]['params']['prompt'] = {'value': 'changed during generation'}
        elif change == 'worker':
            app.instance += '-replacement'
        elif change == 'attempt':
            task['attempt_index'] += 1
        elif change == 'hints':
            task['runtimeHints'] = {'clientRunId': 'changed during generation'}
        elif change == 'declaration':
            # Actual temporary fixture declaration, never a source file.
            adapter.DESCRIPTOR.write_bytes(adapter.DESCRIPTOR.read_bytes() + b' ')
        else:
            task['interrupt_requested'] = True
        return value
    monkeypatch.setattr(handle['model'], 'generate', mutated)
    with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']), pytest.raises(RuntimeError):
        node(model=handle, prompt=prompt, operation='enhance_prompt', enhancement_max_new_tokens=2, min_new_tokens=2)
    assert node.output['proposal_summary'] is None and not node._cache_valid
    assert calls == [True]  # Refusal must be the final drift boundary, not fixture setup.


def test_changed_input_recomputes_with_new_original_producer(actual_pup, monkeypatch):
    app, graph = bound_pair(monkeypatch)
    old = app.node_cache['pup-proposal'].output['proposal_summary']
    app.current_task.update(task_id='changed-input-task', attempt_index=1)
    graph['nodes']['pup-proposal']['params']['prompt']['value'] = 'A blue sphere.'
    owner_tests.dispatch(app, graph, 'pup-proposal')
    new = app.node_cache['pup-proposal'].output['proposal_summary']
    assert new != old and json.loads(new)['producer']['taskId'] == 'changed-input-task'
    assert json.loads(old)['producer']['taskId'] == 'pup-owner-unit'
    assert json.loads(new)['request']['authoredPromptSha256'] == transport.digest_bytes(b'A blue sphere.')


@pytest.mark.parametrize('corruption', ['foreign_class', 'tampered_identity', 'missing_field', 'wrong_type', 'wrong_direction'])
def test_retention_native_forgery_refuses_before_owner_destruction(actual_pup, monkeypatch, corruption):
    app, _ = bound_pair(monkeypatch)
    cached = app.node_cache['pup-proposal']
    if corruption == 'foreign_class':
        app.node_cache['pup-proposal'] = SimpleNamespace(output=cached.output, params={}, _mm_models=[],
            module_name='modules.HuggingFaceTransformers', class_name='GenerateImageVideoText')
    elif corruption == 'tampered_identity':
        cached.module_name = 'foreign'
    else:
        app.modules = deepcopy(app.modules)
        fields = app.modules['modules.HuggingFaceTransformers']['GenerateImageVideoText']['params']
        if corruption == 'missing_field':
            fields.pop('proposal_summary')
        elif corruption == 'wrong_type':
            fields['proposal_summary']['type'] = 'object'
        else:
            fields['proposal_summary']['display'] = 'input'
    ids = set(memory_manager.cache)
    owner = app.node_cache['pup-owner']
    with pytest.raises(ValueError, match='preloaded native'):
        release_owner_caches(app, {'ownerIds': ['pup-owner'], 'nodeIds': ['pup-owner', 'pup-proposal'],
            'retainOutputs': {'pup-proposal': ['proposal_summary']}}, memory_manager)
    assert app.node_cache['pup-owner'] is owner and set(memory_manager.cache) == ids


@pytest.mark.parametrize('kind', ['legacy', 'custom_same_name', 'standard_none'])
def test_unrelated_pure_material_or_none_retention_gains_no_native_identity(kind):
    field = 'text' if kind == 'legacy' else 'proposal_summary'
    value = None if kind == 'standard_none' else 'diagnostic text'
    app = SimpleNamespace(node_cache={'old': SimpleNamespace(output={field: value}, params={}, _mm_models=[])},
        modules={}, _release_node_modular_components=lambda targets: 0,
        _best_effort_device_cache_clear=lambda: [], _best_effort_allocator_trim=lambda: (False, []))
    release_owner_caches(app, {'ownerIds': ['old'], 'nodeIds': ['old'], 'retainOutputs': {'old': [field]}},
                         SimpleNamespace(cache={}))
    assert app.node_cache['old'].output == {field: value}
    assert not hasattr(app.node_cache['old'], 'module_name') and not hasattr(app.node_cache['old'], 'class_name')


def test_new_output_opt_in_is_exact_role_and_source_direction():
    import modules
    from modiff.studio_execution_specs import _execution_spec_role_params
    key = 'modules.HuggingFaceTransformers.GenerateImageVideoText'
    definition = modules.MODULE_MAP['modules.HuggingFaceTransformers']['GenerateImageVideoText']
    public = {'executionPath': 'transformers', 'roles': [('first', key, 0, 0), ('second', key, 1, 0)],
              'bindings': [('second', 'proposal_summary', 'fakeInput')],
              'edges': [('first', 'proposal_summary', 'viewer', 'text'), ('other', 'value', 'second', 'proposal_summary')]}
    first = _execution_spec_role_params(public, key, definition, 'first')
    second = _execution_spec_role_params(public, key, definition, 'second')
    assert first['proposal_summary'] == definition['params']['proposal_summary']
    assert 'proposal_summary' not in second and 'proposal_summary' not in _execution_spec_role_params(public, key, definition)
    assert 'proposal_diagnostic' not in first and 'proposal_diagnostic' not in second
    assert 'proposal_summary' not in nodes.prompt_enhancement_operation_schema()[0]
    assert 'proposal_diagnostic' not in nodes.prompt_enhancement_operation_schema()[0]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), {'opaque': object()}, {'bad': chr(0xD800)},
    {str(index): 0 for index in range(33)}, [[[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]]], [0] * 8192])
def test_transport_refuses_non_json_unicode_depth_entries_and_keys(value):
    with pytest.raises(ValueError):
        transport.bounded_json(value)


def test_exact_utf8_wire_boundary_and_authored_hash_domain():
    cap = transport.MAX_BYTES
    assert len(transport.bounded_json('x' * (cap - 2)).encode()) == cap
    with pytest.raises(ValueError):
        transport.bounded_json('x' * (cap - 1))
    astral = chr(0x1F600)
    assert len(transport.bounded_json(astral).encode()) == 6
    prompt = '  cube ' + astral + '  '
    assert transport.authored_request(prompt, None) == (transport.digest_bytes(prompt.encode()), 'none', None)
    assert transport.authored_request(prompt, None)[0] != transport.digest_bytes(prompt.strip().encode())
    with pytest.raises(ValueError):
        transport.authored_request('x' * 65537, None)
    with pytest.raises(ValueError):
        transport.authored_request('A cube.', '{}')


def test_consumed_and_model_receipt_mutation_refusals(actual_pup, monkeypatch):
    app, _ = bound_pair(monkeypatch)
    value = json.loads(app.node_cache['pup-proposal'].output['proposal_summary'])
    contract = json.loads(adapter.DESCRIPTOR.read_bytes())
    controls = value['request']['controls']
    consumed = value['consumedInput']
    mutants = []
    for key, changed in [('schemaVersion', True), ('renderedPrompt', consumed['renderedPrompt'] + ' '),
        ('processedImageGrid', [[1, 1, 1]])]:
        row = deepcopy(consumed); row[key] = changed; mutants.append(row)
    for key, changed in [('rendered_prompt_utf8_bytes', 1), ('rendered_input_ids_sha256', '0' * 64),
        ('reference_source_sha256', ['sha256:' + '0' * 64]), ('system_prompt_sha256', 'sha256:' + '0' * 64),
        ('operation', 'standard'), ('enhancement_max_new_tokens', True)]:
        row = deepcopy(consumed); row['fingerprints'][key] = changed; mutants.append(row)
    row = deepcopy(consumed); row['unknown'] = None; mutants.append(row)
    for row in mutants:
        with pytest.raises(ValueError):
            transport._consumed(row, controls, contract, 't2i')
    for key, changed in [('source', {'kind': 'local', 'path': 'not allowed'}), ('receiptDigest', '0' * 64),
                         ('extra', None)]:
        receipt = deepcopy(value['modelReceipt']); receipt[key] = changed
        with pytest.raises(ValueError):
            transport._model_receipt(receipt, contract)
    receipt = deepcopy(value['modelReceipt']); receipt['runtime']['quantization'] = {'mode': 'bnb_4bit'}
    with pytest.raises(ValueError):
        transport._model_receipt(receipt, contract)


@pytest.mark.parametrize('valid', [True, False])
def test_bound_structured_summary_never_enters_outer_text(actual_pup, monkeypatch, valid):
    import test_prompt_enhancement_output as outputs
    handle, _ = actual_pup
    row = deepcopy(outputs.ROW)
    if not valid:
        row['bbox'][0] += 1
    suffix = 'A wooden cube <cube_1>. ' + json.dumps([row])
    outputs.substitute_generated_suffix(monkeypatch, handle, suffix)
    node, graph, task, app, _prompt, _layout = context(monkeypatch)
    with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']):
        node(model=handle, prompt='A cube <cube_1>.', layout_conditioning=outputs.LAYOUT,
            operation='enhance_prompt', enhancement_max_new_tokens=256)
    result = node.output['result']; value = json.loads(node.output['proposal_summary'])
    assert node.output['text'] is None and result['text'] == suffix
    assert value['proposal']['text'] is None
    assert value['proposal']['layoutOutputValidated'] is valid and value['proposal']['applicationAllowed'] is valid
    assert (value['proposal']['structuredProposal'] is not None) is valid
    assert (value['proposal']['applicationBlockedReason'] is None) is valid
    assert value['request']['authoredLayoutKind'] == 'object' and value['producer']['recipe'] is None
    assert value['proposal']['applyRequired'] is True and value['proposal']['semanticFidelityVerified'] is False



def test_early_stop_precedes_producer_and_preprocessing_failure(actual_pup, monkeypatch):
    handle, _ = actual_pup
    node, graph, task, app, prompt, _layout = context(monkeypatch)
    task['interrupt_requested'] = True
    app.instance = object()  # Producer capture would fail if entered first.
    normalize = Mock(side_effect=AssertionError('Stopped task reached normalization'))
    model = Mock(side_effect=AssertionError('Stopped task reached model'))
    monkeypatch.setattr(nodes, '_normalized_optional_images', normalize)
    monkeypatch.setattr(handle['model'], 'generate', model)
    with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']), pytest.raises(RuntimeError) as caught:
        node(model=handle, prompt=prompt, operation='enhance_prompt', enhancement_max_new_tokens=2)
    assert isinstance(caught.value.__cause__, InterruptedError)
    assert not normalize.called and not model.called and node.output['proposal_summary'] is None


def test_actual_ordered_reference_summary_and_shape_integrity(actual_pup, monkeypatch):
    from PIL import Image
    import test_prompt_enhancement_phase2 as phase_two
    handle, _ = actual_pup
    images = [Image.new('RGB', (256, 256), color) for color in ('red', 'blue')]
    expected_hashes = [phase_two.rgb_fingerprint(image) for image in images]
    node, graph, task, app, prompt, _layout = context(monkeypatch)
    with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']):
        node(model=handle, prompt=prompt, images=images, operation='enhance_prompt', enhancement_max_new_tokens=2, min_new_tokens=2)
    value = json.loads(node.output['proposal_summary'])
    consumed, controls = value['consumedInput'], value['request']['controls']
    assert consumed['fingerprints']['reference_source_sha256'] == expected_hashes
    assert [row['sha256'] for row in consumed['references']] == expected_hashes
    assert value['proposal']['mode'] == 'i2i' and value['proposal']['layoutOutputValidated'] is None
    contract = json.loads(adapter.DESCRIPTOR.read_bytes())
    mutants = []
    row = deepcopy(consumed); row['references'].reverse(); mutants.append(row)
    row = deepcopy(consumed); row['references'][0]['width'] = True; mutants.append(row)
    row = deepcopy(consumed); row['processedImageGrid'][0][0] = 0; mutants.append(row)
    row = deepcopy(consumed); row['fingerprints']['processed_pixel_shape'][0] += 1; mutants.append(row)
    row = deepcopy(consumed); row['fingerprints']['processed_grid_shape'][0] = True; mutants.append(row)
    for row in mutants:
        with pytest.raises(ValueError):
            transport._consumed(row, controls, contract, 'i2i')
    with pytest.raises(ValueError):
        transport._consumed(consumed, controls, contract, 't2i')


@pytest.mark.parametrize('field', ['trustRemoteCode', 'localFilesOnly', 'safetensorsOnly', 'hostedInference'])
def test_resealed_integer_security_flags_are_not_boolean_contracts(field):
    contract = json.loads(adapter.DESCRIPTOR.read_bytes())
    receipt = {'schemaVersion': 1, 'library': 'transformers', 'task': 'image-video-to-text',
        'source': {'kind': 'hub', 'repository': contract['repository'], 'revision': contract['revision']},
        'loader': {'preprocessorAutoClass': 'AutoProcessor', 'modelAutoClass': 'AutoModelForImageTextToText'},
        'security': deepcopy(nodes.SECURITY_CONTRACT),
        'runtime': {'transformersVersion': contract['transformersVersion'], 'dtype': 'float32', 'device': 'cpu',
            'preprocessorClass': contract['processorClass'], 'modelClass': contract['modelClass'],
            'modelType': None, 'architectures': [], 'quantization': {'mode': 'none'}}}
    receipt['security'][field] = int(receipt['security'][field])
    receipt = nodes._sealed_receipt(receipt)
    assert nodes._receipt_digest_is_valid(receipt)
    with pytest.raises(ValueError):
        transport._model_receipt(receipt, contract)



def test_retention_never_imports_or_resolves_a_lazy_native_claim(monkeypatch):
    import sys
    from types import ModuleType
    from modiff.workflow_auto_lifecycle import _retained_proposal_identity
    calls = []
    preloaded = ModuleType('modules.HuggingFaceTransformers.main')
    def forbidden(name):
        calls.append(name)
        raise AssertionError('Retention invoked dynamic module lookup')
    preloaded.__getattr__ = forbidden
    monkeypatch.setitem(sys.modules, preloaded.__name__, preloaded)
    forged = SimpleNamespace(module_name='modules.HuggingFaceTransformers', class_name='GenerateImageVideoText')
    with pytest.raises(ValueError, match='preloaded native'):
        _retained_proposal_identity(SimpleNamespace(modules={}), forged, {'proposal_summary': '{}'})
    assert calls == []



def test_unrelated_same_name_ndarray_keeps_historical_material_retention():
    import numpy as np
    value = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    app = SimpleNamespace(node_cache={'custom': SimpleNamespace(output={'proposal_summary': value}, params={}, _mm_models=[])},
        modules={}, _release_node_modular_components=lambda targets: 0,
        _best_effort_device_cache_clear=lambda: [], _best_effort_allocator_trim=lambda: (False, []))
    release_owner_caches(app, {'ownerIds': ['custom'], 'nodeIds': ['custom'],
        'retainOutputs': {'custom': ['proposal_summary']}}, SimpleNamespace(cache={}))
    holder = app.node_cache['custom']
    assert np.array_equal(holder.output['proposal_summary'], value)
    assert holder.output['proposal_summary'] is not value
    assert not hasattr(holder, 'module_name') and not hasattr(holder, 'class_name')


@pytest.mark.parametrize('claim', [False, True])
def test_retained_identity_never_calls_custom_node_lookup(claim):
    from modiff.workflow_auto_lifecycle import _retained_proposal_identity
    calls = []
    class Custom:
        __slots__ = ()
        if claim:
            module_name = 'modules.HuggingFaceTransformers'
            class_name = 'GenerateImageVideoText'
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('Retention invoked a custom attribute lookup')
    if claim:
        with pytest.raises(ValueError, match='preloaded native'):
            _retained_proposal_identity(SimpleNamespace(modules={}), Custom(), {'proposal_summary': '{}'})
    else:
        assert _retained_proposal_identity(SimpleNamespace(modules={}), Custom(), {'proposal_summary': '{}'}) == {}
    assert calls == []



def synthetic_consumed_image():
    # Deliberately synthetic primitive projection for strict DTO refusal tests.
    rendered = 'unit prompt'
    contract = json.loads(adapter.DESCRIPTOR.read_bytes())
    controls = {'max_new_tokens': 2, 'min_new_tokens': 0, 'do_sample': False, 'num_beams': 1,
                'repetition_penalty': 1.0, 'return_dict_in_generate': False, 'use_chat_template': True}
    fingerprint = 'sha256:' + '0' * 64
    value = {'schemaVersion': 1, 'renderedPrompt': rendered,
        'references': [{'index': 0, 'mode': 'RGB', 'width': 1, 'height': 1, 'sha256': fingerprint}],
        'processedImageGrid': [[1, 2, 2]], 'fingerprints': {'operation': 'enhance_prompt',
            'enhancement_max_new_tokens': 2, 'prompt_adapter_id': contract['id'], 'system_prompt_mode': 'i2i',
            'system_prompt_sha256': '0' * 64, 'rendered_prompt_sha256': transport.digest_bytes(rendered.encode()),
            'rendered_prompt_utf8_bytes': len(rendered.encode()), 'rendered_input_ids_sha256': fingerprint,
            'reference_source_sha256': [fingerprint], 'processed_pixel_sha256': fingerprint,
            'processed_grid_sha256': fingerprint, 'processed_pixel_shape': [4, 12], 'processed_grid_shape': [1, 3]}}
    return value, controls, contract


@pytest.mark.parametrize('side', ['width', 'height'])
def test_summary_reference_sides_match_actual_normalizer_limit(side):
    value, controls, contract = synthetic_consumed_image()
    value['references'][0][side] = nodes.MAX_MEDIA_SIDE
    transport._consumed(value, controls, contract, 'i2i')
    value['references'][0][side] += 1
    with pytest.raises(ValueError):
        transport._consumed(value, controls, contract, 'i2i')


@pytest.mark.parametrize('codepoint', [0x7f, 0x85, 0x9f])
def test_transport_human_text_rejects_c1_controls(codepoint):
    assert transport.text('caption\t\n\r') == 'caption\t\n\r'
    with pytest.raises(ValueError):
        transport.text('caption' + chr(codepoint))


def test_existing_ascii_consumed_guard_is_additive_to_utf8_wire_budget():
    value, controls, contract = synthetic_consumed_image()
    rendered = chr(0x4e00) * 44000
    value['renderedPrompt'] = rendered
    value['fingerprints']['rendered_prompt_sha256'] = transport.digest_bytes(rendered.encode())
    value['fingerprints']['rendered_prompt_utf8_bytes'] = len(rendered.encode())
    assert len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()) < 256 * 1024
    assert len(json.dumps(value, ensure_ascii=True).encode()) > 256 * 1024
    with pytest.raises(ValueError, match='existing consumed'):
        transport._consumed(value, controls, contract, 'i2i')


def metadata_response(app, *, node='pup-proposal', field='proposal_summary', query='with_metadata=1', index=None):
    matches = {'node': node, 'field': field}
    suffix = '/' + index if index is not None else ''
    if index is not None:
        matches['index'] = index
    request = make_mocked_request('GET', '/cache/' + node + '/' + field + suffix + '?' + query,
        match_info=matches)
    return asyncio.run(app.cache(request))


def rejected_diagnostic_pair(actual_pup, monkeypatch):
    import test_prompt_enhancement_output as outputs
    app, graph = bound_pair(monkeypatch)
    handle = app.node_cache['pup-owner'].output['model']
    # The real node/processor/dispatch remains in use. Only the generated suffix
    # is substituted to reproduce the pre-ID table-count rejection precisely.
    suffix = 'A cube <cube_1>. []'
    outputs.substitute_generated_suffix(monkeypatch, handle, suffix)
    params = graph['nodes']['pup-proposal']['params']
    params.update(prompt={'value': 'A cube <cube_1>.'},
        layout_conditioning={'value': deepcopy(outputs.LAYOUT)},
        enhancement_max_new_tokens={'value': 256}, min_new_tokens={'value': 0})
    owner_tests.dispatch(app, graph, 'pup-proposal')
    return app, suffix


def test_rejected_structured_diagnostic_preserves_exact_text_and_observation(actual_pup, monkeypatch):
    app, suffix = rejected_diagnostic_pair(actual_pup, monkeypatch)
    node = app.node_cache['pup-proposal']
    summary = node.output['proposal_summary']
    parsed = json.loads(summary)
    assert node.output['text'] is None and node.output['result']['text'] == suffix
    assert parsed['proposal']['text'] is None and parsed['proposal']['structuredProposal'] is None
    assert parsed['proposal']['applicationAllowed'] is False
    assert parsed['proposal']['applicationBlockedReason'].endswith('exactly the authored target IDs.')
    response = metadata_response(app, field='proposal_diagnostic')
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload['value'] == suffix == node.output['proposal_diagnostic']
    assert payload['observation']['diagnosticSha256'] == transport.digest_bytes(suffix.encode('utf-8'))
    assert payload['observation']['summarySha256'] == transport.digest_bytes(summary.encode('utf-8'))
    assert payload['observation']['producer']['taskId'] == app.current_task['task_id']
    assert payload['observation']['dispatch']['status'] == 'succeeded'
    before = dict(app._prompt_summary_observations)
    assert metadata_response(app, field='proposal_diagnostic').body == response.body
    assert app._prompt_summary_observations == before and node.output['proposal_summary'] == summary
    assert cache_response(app, 'proposal_diagnostic').status == 400
    assert cache_response(app, 'result').status == 400


def terminal_diagnostic_preview(actual_pup, tmp_path, monkeypatch, *, tamper=None):
    """Real Preview lease and terminal release; only generated suffix is data."""
    import weakref
    import torch
    import modules
    import test_ancillary_execution as ancillary
    import test_ancillary_owner_lifecycle as lifecycle
    import test_prompt_enhancement_output as outputs
    handle, _ = actual_pup
    suffix = 'A cube <cube_1>. []'
    encoded = handle['preprocessor'].tokenizer(suffix, return_tensors='pt',
        add_special_tokens=False)['input_ids']
    def generated(_model, **kwargs):
        return torch.cat([kwargs['input_ids'], encoded.to(kwargs['input_ids'].device)], dim=1)
    # An unbound class method avoids retaining the soon-to-be-released model
    # through the monkeypatch undo stack or a closure over its handle.
    monkeypatch.setattr(type(handle['model']), 'generate', generated)
    memory_manager.remove(handle['model'])
    del handle
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP, index=4)
    graph['runtimeHints']['resourceMode'] = 'expert'
    graph['nodes']['provider']['params']['device'] = {'value': stage_one.DEFAULT_DEVICE}
    graph['nodes']['proposal']['params'].update(prompt={'value': 'A cube <cube_1>.'},
        layout_conditioning={'value': deepcopy(outputs.LAYOUT)},
        enhancement_max_new_tokens={'value': 256}, min_new_tokens={'value': 0})
    app = lifecycle.app_for(tmp_path, monkeypatch, graph)
    app._prepare_auto_runtime_for_graph = lambda _hints: None
    app._best_effort_device_cache_clear = lambda: []
    app._best_effort_allocator_trim = lambda: (False, [])
    captured = {}
    def after(node_id):
        node = app.node_cache[node_id]
        if node_id == 'provider':
            captured['model'] = weakref.ref(node.output['model']['model'])
        elif node_id == 'proposal':
            captured['proposal'] = weakref.ref(node)
            captured['summary'] = node.output['proposal_summary']
            captured['diagnostic'] = node.output['proposal_diagnostic']
            if tamper is not None:
                node.output['proposal_diagnostic'] = tamper(suffix)
    lifecycle.quiet_dispatch(app, after)
    return app, graph, captured, suffix


def test_rejected_diagnostic_survives_actual_preview_terminal_release(actual_pup, tmp_path, monkeypatch):
    import test_ancillary_owner_lifecycle as lifecycle
    from modiff.workflow_auto_lifecycle import _RetainedPromptProposal
    app, graph, captured, suffix = terminal_diagnostic_preview(actual_pup, tmp_path, monkeypatch)
    lifecycle.leased(app, graph)
    assert 'provider' not in app.node_cache and not memory_manager.cache
    assert captured['model']() is None and captured['proposal']() is None
    holder = app.node_cache['proposal']
    assert type(holder) is _RetainedPromptProposal
    assert holder.output['text'] is None and captured['diagnostic'] == suffix
    summary = json.loads(holder.output['proposal_summary'])
    assert summary['proposal']['applicationAllowed'] is False
    assert summary['proposal']['structuredProposal'] is None
    assert summary['proposal']['applicationBlockedReason'].endswith('exactly the authored target IDs.')
    response = metadata_response(app, node='proposal', field='proposal_diagnostic')
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload['value'] == suffix == holder.output['proposal_diagnostic']
    assert payload['observation']['diagnosticSha256'] == transport.digest_bytes(suffix.encode('utf-8'))
    assert payload['observation']['summarySha256'] == transport.digest_bytes(captured['summary'].encode('utf-8'))
    assert payload['observation']['producer']['taskId'] == app.current_task['task_id']
    assert payload['observation']['dispatch']['status'] == 'succeeded'
    records = dict(app._prompt_summary_observations)
    assert metadata_response(app, node='proposal', field='proposal_diagnostic').body == response.body
    assert app._prompt_summary_observations == records
    assert metadata_response(app, node='proposal').status == 200
    request = make_mocked_request('GET', '/cache/proposal/result',
        match_info={'node': 'proposal', 'field': 'result'})
    assert asyncio.run(app.cache(request)).status == 400
    holder.output['proposal_diagnostic'] = suffix + ' '
    assert metadata_response(app, node='proposal', field='proposal_diagnostic').status == 409


@pytest.mark.parametrize('omit_legacy_field', [False, True])
def test_plain_preview_terminal_release_keeps_legacy_outputs_without_diagnostic(
        actual_pup, tmp_path, monkeypatch, omit_legacy_field):
    import test_ancillary_owner_lifecycle as lifecycle
    app, graph = lifecycle.cold_preview(actual_pup, tmp_path, monkeypatch)
    app._prepare_auto_runtime_for_graph = lambda _hints: None
    def after(node_id):
        if node_id == 'proposal':
            assert app.node_cache[node_id].output['proposal_diagnostic'] is None
            if omit_legacy_field:
                app.node_cache[node_id].output.pop('proposal_diagnostic')
    lifecycle.quiet_dispatch(app, after)
    lifecycle.leased(app, graph)
    assert 'provider' not in app.node_cache and not memory_manager.cache
    assert set(app.node_cache['proposal'].output) == {'proposal_summary', 'text', 'result'}
    assert metadata_response(app, node='proposal').status == 200
    assert metadata_response(app, node='proposal', field='proposal_diagnostic').status == 404


@pytest.mark.parametrize('mutation', ['changed', 'opaque', 'oversized'])
def test_preview_terminal_release_rejects_changed_or_unbounded_diagnostic_before_removal(
        actual_pup, tmp_path, monkeypatch, mutation):
    import test_ancillary_owner_lifecycle as lifecycle
    def tamper(suffix):
        if mutation == 'changed':
            return suffix + ' '
        if mutation == 'opaque':
            return {'text': suffix}
        return 'x' * (transport.DIAGNOSTIC_MAX_BYTES + 1)
    app, graph, captured, _suffix = terminal_diagnostic_preview(actual_pup, tmp_path, monkeypatch, tamper=tamper)
    with pytest.raises(ValueError, match='diagnostic|exact string contract'):
        lifecycle.leased(app, graph)
    assert 'provider' in app.node_cache and memory_manager.cache
    assert captured['model']() is not None and captured['proposal']() is app.node_cache['proposal']


@pytest.mark.parametrize('change', ['diagnostic', 'opaque', 'summary', 'output', 'worker', 'input', 'registry', 'alias'])
def test_rejected_diagnostic_never_bypasses_native_observation_seals(actual_pup, monkeypatch, change):
    app, suffix = rejected_diagnostic_pair(actual_pup, monkeypatch)
    node = app.node_cache['pup-proposal']
    assert metadata_response(app, field='proposal_diagnostic').status == 200
    if change == 'diagnostic':
        node.output['proposal_diagnostic'] = suffix + ' '
    elif change == 'opaque':
        node.output['proposal_diagnostic'] = {'text': suffix}
    elif change == 'summary':
        node.output['proposal_summary'] += ' '
    elif change == 'output':
        node.output = dict(node.output)
    elif change == 'worker':
        app.instance += '-replacement'
    elif change == 'input':
        node._cache_input_snapshot = ('changed',)
    elif change == 'registry':
        field = app.modules[transport.NODE_MODULE][transport.NODE_ACTION]['params']['proposal_diagnostic']
        monkeypatch.setitem(field, 'type', 'object')
    elif change == 'alias':
        app.node_cache['renamed'] = node
        app._prompt_summary_observations['renamed'] = app._prompt_summary_observations['pup-proposal']
        assert metadata_response(app, node='renamed', field='proposal_diagnostic').status == 409
        return
    assert metadata_response(app, field='proposal_diagnostic').status in (400, 409)


@pytest.mark.parametrize('value', ['', None, object(), 'x' * 262145, chr(0xD800)])
def test_diagnostic_omits_unbounded_or_nonunicode_output_without_truncation(value):
    result = {'systemPromptMode': 't2i_bbox', 'applicationAllowed': False, 'text': value}
    assert transport.diagnostic_output(result) is None
    assert result['text'] is value


def test_diagnostic_scope_and_exact_unicode_byte_boundary():
    result = {'systemPromptMode': 'i2i_bbox', 'applicationAllowed': False, 'text': '\U0001f600' * 262144}
    assert transport.diagnostic_output(result) == result['text']
    assert len(result['text'].encode('utf-8')) == transport.DIAGNOSTIC_MAX_BYTES
    result.update(applicationAllowed=True)
    assert transport.diagnostic_output(result) is None
    result.update(applicationAllowed=False, systemPromptMode='i2i')
    assert transport.diagnostic_output(result) is None


def test_diagnostic_missing_plain_unobserved_and_deleted_outputs_remain_closed(actual_pup, monkeypatch):
    app, _ = bound_pair(monkeypatch)
    assert metadata_response(app, field='proposal_diagnostic').status == 404
    app, _ = rejected_diagnostic_pair(actual_pup, monkeypatch)
    for query in ['with_metadata=0', 'with_metadata=1&unknown=1', 'with_metadata=1&with_metadata=1']:
        assert metadata_response(app, field='proposal_diagnostic', query=query).status == 400
    assert metadata_response(app, field='proposal_diagnostic', index='0').status == 400
    records = app._prompt_summary_observations
    app._prompt_summary_observations = {}
    assert metadata_response(app, field='proposal_diagnostic').status == 409
    app._prompt_summary_observations = records
    assert app._release_cached_nodes(['pup-proposal'])
    assert metadata_response(app, field='proposal_diagnostic').status == 404


def test_native_cache_metadata_is_independent_preoutput_observation(actual_pup, monkeypatch):
    app, _ = bound_pair(monkeypatch)
    node = app.node_cache['pup-proposal']
    before_nodes = dict(app.node_cache)
    first = metadata_response(app)
    assert first.status == 200
    payload = json.loads(first.body)
    assert set(payload) == {'value', 'observation'}, 'native metadata must be an independently minted envelope'
    assert payload['value'] == node.output['proposal_summary']
    observation = payload['observation']
    assert observation['summarySha256'] == transport.digest_bytes(payload['value'].encode('utf-8'))
    assert observation['producer']['taskId'] == 'pup-owner-unit'
    assert observation['dispatch']['taskId'] == 'pup-owner-unit'
    assert observation['dispatch']['status'] == 'succeeded'
    assert observation['producer']['recipe'] is None and 'recipe' not in observation['dispatch']
    assert observation['request']['authoredPromptSha256'] == transport.digest_bytes(b'A red cube.')
    assert observation['consumedInput']['references'] == []
    assert 'renderedPrompt' not in observation['consumedInput']
    assert app.node_cache == before_nodes
    records = dict(app._prompt_summary_observations)
    second = metadata_response(app)
    assert second.body == first.body and app._prompt_summary_observations == records
    assert cache_response(app, 'proposal_summary').body.decode('utf-8') == payload['value']
    persist_metadata('metadata-cold', app, first)


def test_warm_metadata_keeps_original_A_and_actual_dispatch_B(actual_pup, monkeypatch):
    hints = {'clientRunId': 'A', 'workflowCanvasEpoch': 1}
    app, graph = bound_pair(monkeypatch, hints=hints)
    before = json.loads(metadata_response(app).body)
    hints.update(clientRunId='B', workflowCanvasEpoch=2)
    app.current_task.update(task_id='task-B', attempt_index=1)
    owner_tests.dispatch(app, graph, 'pup-owner')
    owner_tests.dispatch(app, graph, 'pup-proposal')
    after = json.loads(metadata_response(app).body)
    assert app.node_cache['pup-proposal']._has_changed is False
    assert after['value'] == before['value']
    assert after['observation']['producer'] == before['observation']['producer']
    assert after['observation']['producer']['taskId'] == 'pup-owner-unit'
    assert after['observation']['dispatch']['taskId'] == 'task-B'
    assert after['observation']['dispatch']['attemptIndex'] == 1
    assert after['observation']['dispatch']['uiContext']['clientRunId'] == 'B'
    assert after['observation']['dispatch']['uiContext']['workflowCanvasEpoch'] == 2
    assert after['observation']['dispatch']['status'] == 'cached'
    assert after['observation']['producer']['graphHash'] == after['observation']['dispatch']['graphHash']
    persist_metadata('metadata-warm', app, metadata_response(app))


def test_native_metadata_rejects_invalid_options_without_mutation(actual_pup, monkeypatch):
    app, _ = bound_pair(monkeypatch)
    before = dict(app._prompt_summary_observations)
    for query, field, index in [('with_metadata=0', 'proposal_summary', None),
        ('with_metadata=', 'proposal_summary', None),
        ('with_metadata=1&with_metadata=1', 'proposal_summary', None),
        ('with_metadata=1&filename=sidecar', 'proposal_summary', None),
        ('with_metadata=1&unknown=1', 'proposal_summary', None),
        ('with_metadata=1', 'text', None), ('with_metadata=1', 'proposal_summary', '0')]:
        assert metadata_response(app, query=query, field=field, index=index).status == 400
    assert app._prompt_summary_observations == before


@pytest.mark.parametrize('change', ['summary', 'output', 'worker', 'source', 'invalidated', 'input', 'sources', 'native'])
def test_native_metadata_never_accepts_replaced_or_stale_cache(actual_pup, monkeypatch, change):
    app, _ = bound_pair(monkeypatch)
    node = app.node_cache['pup-proposal']
    assert metadata_response(app).status == 200
    if change == 'summary':
        forged = json.loads(node.output['proposal_summary'])
        forged['producer']['taskId'] = 'forged-new-producer'
        node.output['proposal_summary'] = json.dumps(forged, ensure_ascii=False)
    elif change == 'output':
        node.output = dict(node.output)
    elif change == 'worker':
        app.instance += '-replacement'
    elif change == 'source':
        app.backend_source_identity = {**app.backend_source_identity,
            'fingerprint': 'sha256:backend-source-v1:' + 'f' * 64}
    elif change == 'invalidated':
        node.invalidate_cache()
    elif change == 'input':
        node._cache_input_snapshot = ('replaced',)
    elif change == 'sources':
        node._cache_input_sources = frozenset({'foreign-owner'})
    elif change == 'native':
        app.node_cache['pup-proposal'] = SimpleNamespace(output=dict(node.output), params={},
            module_name=transport.NODE_MODULE, class_name=transport.NODE_ACTION)
    assert metadata_response(app).status in (400, 409)


def test_native_metadata_unknown_node_is_not_found(actual_pup, monkeypatch):
    app, _ = bound_pair(monkeypatch)
    assert metadata_response(app, node='unknown').status == 404


def test_same_bytes_recompute_mints_new_incarnation(actual_pup, monkeypatch):
    app, graph = bound_pair(monkeypatch)
    old = app._prompt_summary_observations['pup-proposal']
    summary = app.node_cache['pup-proposal'].output['proposal_summary']
    app.node_cache['pup-proposal'].invalidate_cache()
    owner_tests.dispatch(app, graph, 'pup-proposal')
    assert app.node_cache['pup-proposal']._has_changed is True
    assert app.node_cache['pup-proposal'].output['proposal_summary'] == summary
    current = app._prompt_summary_observations['pup-proposal']
    assert current.output_token() is not old.output_token()
    assert current.native_node() is app.node_cache['pup-proposal']
    assert json.loads(metadata_response(app).body)['observation']['dispatch']['status'] == 'succeeded'


def test_native_metadata_uses_actual_pre_generation_ordered_RGB_not_file_hash(actual_pup, monkeypatch):
    from PIL import Image
    from modiff.config import CONFIG
    import test_prompt_enhancement_phase2 as phase_two
    app, graph = bound_pair(monkeypatch)
    root = Path(CONFIG.paths['images']) / 'sidecar-native-reference-unit'
    root.mkdir(parents=True)
    images = [Image.new('RGB', (256, 256), color) for color in ('red', 'blue')]
    files = [root / (str(index) + '.png') for index in range(2)]
    for path, image in zip(files, images):
        image.save(path)
    expected = [phase_two.rgb_fingerprint(image) for image in images]
    graph['nodes']['references'] = {'module': 'modules.Image', 'action': 'Load',
        'params': {'file': {'value': [str(path) for path in files]}, 'alpha_channel': {'value': 'ignore'}}}
    graph['nodes']['pup-proposal']['params']['images'] = {'sourceId': 'references', 'sourceKey': 'image'}
    graph['paths'] = [['pup-owner', 'references', 'pup-proposal']]
    app._active_graph_node_ids = set(graph['nodes'])
    owner_tests.dispatch(app, graph, 'references')
    owner_tests.dispatch(app, graph, 'pup-proposal')
    response = metadata_response(app)
    assert response.status == 200
    observed = json.loads(response.body)['observation']['consumedInput']
    assert [row['index'] for row in observed['references']] == [0, 1]
    assert [row['sha256'] for row in observed['references']] == expected
    assert [row['width'] for row in observed['references']] == [256, 256]
    assert [row['height'] for row in observed['references']] == [256, 256]
    assert observed['fingerprints']['reference_source_sha256'] == expected
    assert expected != ['sha256:' + hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
    assert len(observed['processedImageGrid']) == 2
    assert 'renderedPrompt' not in observed and not {'text', 'pixel_values'} & set(observed)


def test_native_metadata_pre_generation_observation_rejects_alias_mutation(actual_pup, monkeypatch):
    handle, _ = actual_pup
    original_projection = adapter._consumed_projection
    original_generate = handle['model'].generate
    captured = {}
    def projection(*args, **kwargs):
        fields, consumed = original_projection(*args, **kwargs)
        captured['consumed'] = consumed
        return fields, consumed
    def mutate_after_capture(**kwargs):
        # Valid digest syntax cannot turn a later mutated source observation
        # into the immutable actual pre-generation observation.
        captured['consumed']['fingerprints']['rendered_input_ids_sha256'] = 'sha256:' + '0' * 64
        return original_generate(**kwargs)
    monkeypatch.setattr(adapter, '_consumed_projection', projection)
    monkeypatch.setattr(handle['model'], 'generate', mutate_after_capture)
    node, graph, task, app, prompt, _ = context(monkeypatch)
    with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']), pytest.raises(RuntimeError) as caught:
        node(model=handle, prompt=prompt, operation='enhance_prompt', enhancement_max_new_tokens=2, min_new_tokens=2)
    assert isinstance(caught.value.__cause__, ValueError)
    assert 'normalized input changed' in str(caught.value.__cause__)
    assert node.output['proposal_summary'] is None
    assert '_prompt_observation_pending' not in vars(node)


@pytest.mark.parametrize('fault', ['none', 'allocator', 'survivor'])
def test_native_terminal_observation_rebind_is_success_only_after_real_release(actual_pup, tmp_path, monkeypatch, fault):
    import weakref
    import test_ancillary_owner_lifecycle as lifecycle
    app, graph = lifecycle.cold_preview(actual_pup, tmp_path, monkeypatch)
    app._prepare_auto_runtime_for_graph = lambda _hints: None
    weak = {}
    held = []
    def after(node_id):
        if node_id == 'provider':
            model = app.node_cache[node_id].output['model']['model']
            weak['model'] = weakref.ref(model)
            if fault == 'survivor':
                held.append(model)
        elif node_id == 'proposal':
            weak['proposal'] = weakref.ref(app.node_cache[node_id])
    lifecycle.quiet_dispatch(app, after)
    if fault == 'allocator':
        app._best_effort_allocator_trim = lambda: (False, ['explicit unit trim failure'])
    if fault == 'none':
        lifecycle.leased(app, graph)
    else:
        with pytest.raises(ValueError, match='trim failure|references remain alive'):
            lifecycle.leased(app, graph)
    assert 'provider' not in app.node_cache and not memory_manager.cache
    assert weak['proposal']() is None
    assert (weak['model']() is None) == (fault != 'survivor')
    response = metadata_response(app, node='proposal')
    if fault == 'none':
        assert response.status == 200
        payload = json.loads(response.body)
        assert payload['observation']['producer']['taskId'] == 'explicit-owner-boundary-unit'
        assert payload['observation']['dispatch']['status'] == 'succeeded'
        assert payload['value'] == app.node_cache['proposal'].output['proposal_summary']
    else:
        assert response.status == 409
        # The independent detached raw failure diagnostic is preserved.
        assert app.node_cache['proposal'].output['proposal_summary']
        assert 'proposal' not in app._prompt_summary_observations


def test_native_metadata_custom_properties_are_not_inspected(monkeypatch):
    from modiff.server import WebServer
    touched = []
    class Custom:
        @property
        def output(self):
            touched.append('output')
            raise AssertionError('custom output was inspected')
        @property
        def module_name(self):
            touched.append('module')
            raise AssertionError('custom module was inspected')
        def __getattr__(self, key):
            touched.append(key)
            raise AssertionError('custom attribute was inspected')
    app = object.__new__(WebServer)
    app.node_cache = {'pup-proposal': Custom()}
    assert metadata_response(app).status == 400
    assert touched == [] and '_prompt_summary_observations' not in vars(app)


def test_native_metadata_busy_unobserved_and_registry_changed_are_closed(actual_pup, monkeypatch):
    app, _ = bound_pair(monkeypatch)
    lock = asyncio.Lock()
    asyncio.run(lock.acquire())
    app._node_cache_lock = lock
    before = dict(app._prompt_summary_observations)
    assert metadata_response(app).status == 409
    assert app._prompt_summary_observations == before
    lock.release()
    field = app.modules[transport.NODE_MODULE][transport.NODE_ACTION]['params']['proposal_summary']
    monkeypatch.setitem(field, 'type', 'model')
    assert metadata_response(app).status == 400
    monkeypatch.setitem(field, 'type', 'string')
    app._prompt_summary_observations = {}
    assert metadata_response(app).status == 409
    assert app._prompt_summary_observations == {}
    assert cache_response(app, 'proposal_summary').status == 200


def test_native_metadata_prunes_actual_deletion_and_refuses_alias(actual_pup, monkeypatch):
    app, _ = bound_pair(monkeypatch)
    record = app._prompt_summary_observations['pup-proposal']
    app.node_cache['renamed'] = app.node_cache['pup-proposal']
    app._prompt_summary_observations['renamed'] = record
    assert metadata_response(app, node='renamed').status == 409
    assert app._release_cached_nodes(['pup-proposal'])
    assert 'pup-proposal' not in app._prompt_summary_observations
    # A real remaining unrelated cache alias grants no producer rename.
    assert metadata_response(app, node='renamed').status == 409


def test_native_metadata_does_not_load_neural_or_processor_factories_on_GET(actual_pup, monkeypatch):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    app, _ = bound_pair(monkeypatch)
    forbidden = Mock(side_effect=AssertionError('GET reached a neural/processor factory'))
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', forbidden)
    monkeypatch.setattr(AutoModelForImageTextToText, 'from_pretrained', forbidden)
    assert metadata_response(app).status == 200
    assert not forbidden.called


def test_native_metadata_rechecks_actual_external_processor_metadata_bytes(actual_pup, monkeypatch):
    from modiff.config import CONFIG
    app, _ = bound_pair(monkeypatch)
    assert metadata_response(app).status == 200
    source = (Path(CONFIG.hf['cache_dir']) / ('models--' + stage_one.REPOSITORY.replace('/', '--')) /
        'snapshots' / stage_one.REVISION / 'system_prompts.json')
    source.write_bytes(source.read_bytes() + b'\n')
    assert metadata_response(app).status == 409


def test_native_observation_source_guard_reads_fresh_helper_and_data_bytes(actual_pup, tmp_path, monkeypatch):
    # This is a source-guard unit, using real copied public source/data bytes;
    # it does not relabel a neural cache from another source root.
    from modiff import backend_source_identity as source_identity
    private = tmp_path / 'finite-source-guard'
    (private / 'modiff').mkdir(parents=True)
    (private / 'data').mkdir()
    origin = Path(transport.__file__).resolve().parents[1]
    for source in (origin / 'data').glob('*.json'):
        (private / 'data' / source.name).write_bytes(source.read_bytes())
    for name in ('prompt_enhancement_summary.py', 'backend_source_identity.py'):
        (private / 'modiff' / name).write_bytes((origin / 'modiff' / name).read_bytes())
    monkeypatch.setattr(transport, '__file__', str(private / 'modiff/prompt_enhancement_summary.py'))
    monkeypatch.setattr(source_identity, '__file__', str(private / 'modiff/backend_source_identity.py'))
    first = transport._source_snapshot()
    pending = transport._PendingObservation(
        transport.bounded_json({'workerInstance': 'source-unit',
            'backendSourceFingerprint': json.loads(first)['backend']}), '{}', '{}', 'sha256:' + '0' * 64, first)
    record = SimpleNamespace(pending=pending)
    server = SimpleNamespace(instance='source-unit', backend_source_identity={'fingerprint': json.loads(first)['backend']})
    transport._source_current(server, record)
    helper = private / 'modiff/prompt_enhancement_summary.py'
    helper_bytes = helper.read_bytes()
    helper.write_bytes(helper_bytes + b'\n# actual private helper drift\n')
    with pytest.raises(ValueError, match='changed source'):
        transport._source_current(server, record)
    helper.write_bytes(helper_bytes)
    data = next((private / 'data').glob('*.json'))
    data.write_bytes(data.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed source'):
        transport._source_current(server, record)


def persist_metadata(label, app, response):
    output = os.environ.get('MODIFF_PUP_SUMMARY_FIXTURES')
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        with (path / (label + '.json')).open('x', encoding='utf-8') as stream:
            json.dump({'proofScope': 'Actual normal native tiny CPU dispatcher and cache route; random neural checkpoint, not trained or public eligibility proof',
                'responseBodyUtf8': response.body.decode('utf-8'), 'response': json.loads(response.body),
                'independentPreGenerateExpectations': deepcopy(app._unit_original_expectation),
                'currentDispatchTask': {key: deepcopy(app.current_task.get(key)) for key in ('task_id', 'attempt_index', 'sid', 'runtimeHints')}}, stream, indent=2, ensure_ascii=False)


def test_native_observation_inventory_is_bounded_and_pruned_only_on_mutation():
    import weakref
    class Holder:
        pass
    server = SimpleNamespace(node_cache={}, _prompt_summary_observations={})
    pending = transport._PendingObservation('{}', '{}', '{}', 'sha256:' + '0' * 64, '{}')
    for index in range(transport.MAX_CACHE_OBSERVATIONS + 1):
        holder = Holder()
        holder.output = {}
        token = transport._OutputIncarnation(holder.output)
        holder._prompt_output_incarnation = token
        key = 'synthetic-observation-' + str(index)
        server.node_cache[key] = holder
        server._prompt_summary_observations[key] = transport._CacheObservation(
            pending, weakref.ref(holder), weakref.ref(token), None, None, frozenset(), '{}')
    # Pure inventory control, no native producer/admission claim is made by
    # these deliberately minimal model-free private holder doubles.
    transport.prune_cache_observations(server)
    assert len(server._prompt_summary_observations) == transport.MAX_CACHE_OBSERVATIONS
    assert 'synthetic-observation-0' not in server._prompt_summary_observations
    removed = 'synthetic-observation-1'
    server.node_cache.pop(removed)
    transport.prune_cache_observations(server)
    assert removed not in server._prompt_summary_observations


@pytest.mark.parametrize('limit', ['json', 'bytes', 'entries'])
def test_source_census_stops_at_incremental_bound_before_next_entry(tmp_path, monkeypatch, limit):
    private = tmp_path / 'finite-census'
    data = private / 'data'; data.mkdir(parents=True)
    (private / 'modiff').mkdir()
    monkeypatch.setattr(transport, '__file__', str(private / 'modiff/prompt_enhancement_summary.py'))
    count = {'json': 257, 'bytes': 9, 'entries': 1025}[limit]
    rows = []
    for index in range(count):
        path = data / (str(index) + ('.txt' if limit == 'entries' else '.json'))
        with path.open('wb') as stream:
            if limit == 'bytes':
                stream.truncate(8 * 1024**2)
            else:
                stream.write(b'{}')
        rows.append(path)
    original_iterdir = Path.iterdir
    def bounded_iterator(path):
        if path == data:
            yield from rows
            pytest.fail('The source census continued after its finite inventory bound')
        else:
            yield from original_iterdir(path)
    monkeypatch.setattr(Path, 'iterdir', bounded_iterator)
    with pytest.raises(ValueError, match='finite|bound|inventory'):
        transport._source_snapshot()

@pytest.mark.parametrize('shape', ['small', 'max-file', 'directory-json'])
def test_source_census_retains_plain_control_contract(tmp_path, monkeypatch, shape):
    from modiff import prompt_enhancement as adapter
    private = tmp_path / 'finite-census-control'
    data = private / 'data';data.mkdir(parents=True)
    (private / 'modiff').mkdir()
    monkeypatch.setattr(transport, '__file__', str(private / 'modiff/prompt_enhancement_summary.py'))
    path = data / 'entry.json'
    if shape == 'directory-json':
        path.mkdir()
        with pytest.raises(ValueError, match='plain file'):
            transport._source_snapshot()
        return
    with path.open('wb') as stream:
        stream.write(b'{}')
        if shape == 'max-file':stream.truncate(8 * 1024**2)
    (data / 'ignored').mkdir()
    (data / 'ignored.txt').write_text('metadata-only ignored entry')
    class AfterCensus(Exception): pass
    reached = Mock(side_effect=AfterCensus('unit source census completed'))
    monkeypatch.setattr(adapter, '_declared_contract', reached)
    with pytest.raises(AfterCensus, match='census completed'):
        transport._source_snapshot()
    assert reached.call_count == 1



@pytest.mark.parametrize('tamper', [None, 'nativeText', 'caption', 'description'])
def test_append_summary_independently_recomposes_from_authored_source(actual_pup, monkeypatch, tamper):
    import test_prompt_enhancement_output as outputs
    handle, _ = actual_pup
    suffix = '  Soft light reveals the fine wood grain.\n'
    outputs.substitute_generated_suffix(monkeypatch, handle, suffix)
    node, graph, _task, app, _prompt, _layout = context(monkeypatch)
    authored = '  Keep the cube <cube_1>.\n'
    layout = deepcopy(outputs.LAYOUT)
    original_finish = transport.BoundSummary.finish
    def finish(bound, result, controls):
        if tamper == 'nativeText':
            result['structuredProposal']['nativeText'] += ' forged'
        elif tamper == 'caption':
            result['structuredProposal']['caption'] = 'changed prefix'
        elif tamper == 'description':
            result['structuredProposal']['layoutConditioning']['regions'][0]['description'] = 'changed'
        return original_finish(bound, result, controls)
    monkeypatch.setattr(transport.BoundSummary, 'finish', finish)
    def generate():
        with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']):
            node(model=handle, prompt=authored, layout_conditioning=layout,
                 operation='enhance_prompt_append_details', enhancement_max_new_tokens=256)
    if tamper:
        with pytest.raises(RuntimeError) as caught:
            generate()
        assert isinstance(caught.value.__cause__, ValueError)
        assert 'changed after validation' in str(caught.value.__cause__)
        assert node.output['proposal_summary'] is None
    else:
        generate()
        summary = json.loads(node.output['proposal_summary'])
        proposal = summary['proposal']['structuredProposal']
        assert node.output['text'] is summary['proposal']['text'] is None
        assert proposal == {'schemaVersion': 3, 'kind': 'caption-append-preserve-layout-v1',
            'caption': authored + '\n\n' + suffix, 'layoutConditioning': layout, 'nativeText': suffix}
        assert summary['proposal']['mode'] == 't2i'
        assert summary['consumedInput']['fingerprints']['operation'] == 'enhance_prompt_append_details'
        assert summary['proposal']['layoutOutputValidated'] is True
        assert summary['proposal']['semanticFidelityVerified'] is False
        assert node.output['proposal_diagnostic'] is None


def test_append_rejection_in_plain_native_mode_retains_sealed_diagnostic(actual_pup, monkeypatch):
    import test_prompt_enhancement_output as outputs
    handle, _ = actual_pup
    outputs.substitute_generated_suffix(monkeypatch, handle, 'An extra <unknown> tag.')
    node, graph, _task, app, _prompt, _layout = context(monkeypatch)
    with native_component_graph_scope(graph, app.backend_source_identity['fingerprint']):
        node(model=handle, prompt='A cube <cube_1>.', layout_conditioning=outputs.LAYOUT,
             operation='enhance_prompt_append_details', enhancement_max_new_tokens=256)
    summary = json.loads(node.output['proposal_summary'])
    assert summary['proposal']['mode'] == 't2i' and summary['proposal']['text'] is None
    assert summary['proposal']['applicationAllowed'] is False
    assert summary['proposal']['structuredProposal'] is None
    assert node.output['proposal_diagnostic'] == 'An extra <unknown> tag.'
