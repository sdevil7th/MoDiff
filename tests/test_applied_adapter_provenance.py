"""Applied-adapter observations from ordinary graph dispatch, no downloads.

Tiny CPU PEFT fixtures exercise activation and the existing owner lifecycle.
These checks do not qualify native model output or supported hardware.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import weakref

import pytest

from modiff.applied_adapter_provenance import project_applied_adapters
from modiff.auxiliary_lora import ResolvedLoraDescriptor, resolve_lora_descriptors
from modiff.execution_input_provenance import bind_generation_input_origins, build_resolved_execution_inputs
from test_native_image_disk_lora import (  # noqa: F401 -- register shared fixtures
    TinyPipeline, descriptor, executable_graph, execute_graph, graph_runtime, native_runtime,
)


def test_normal_graph_receipt_records_applied_zero_scale(graph_runtime, tmp_path):
    server, _, _, _, _ = graph_runtime
    graph = executable_graph([descriptor(tmp_path, scale=0.0)])
    assert execute_graph(server, graph) == [[1.0, 2.0]]
    record = server._resolved_input_records['models']
    applied = record['appliedAdapters']
    assert applied['status'] == 'complete'
    assert applied['observation'] == 'activated'
    assert applied['adapters'][0]['scale'] == 0
    assert applied['adapters'][0]['adapterName'] == 'style'


def hub_chain_graph(tmp_path, monkeypatch, scales=(0.75, 1.5), *, name=None):
    """Actual connected Lora nodes backed by temporary managed-cache bytes."""
    from modiff.config import CONFIG
    from utils import huggingface as hf

    cache = Path(CONFIG.hf['cache_dir'])
    monkeypatch.setattr(hf, '_hf_cache_locations', lambda: [('Configured', str(cache))])
    monkeypatch.setattr(hf, 'cached_file_path', lambda repo, filename, *, revision, **kw:
                        str(cache / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision / filename))
    graph = executable_graph()
    order, previous = [], None
    for index, scale in enumerate(scales):
        local = descriptor(tmp_path, index + 1.0)
        weight = Path(local['artifact']['root']) / local['artifact']['weight_name']
        repo, revision, alias = f'example/style-{index}', 'a' * 40, 'weights/adapter.safetensors'
        installed = cache / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision / alias
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_bytes(weight.read_bytes())
        node_id = f'style-{index}'
        params = {key: {'value': value} for key, value in dict(
            model={'source': 'hub', 'value': repo}, revision=revision, weight_name=alias,
            expected_sha256=hashlib.sha256(installed.read_bytes()).hexdigest(),
            adapter_name=name or f'style_{index}', scale=scale, scheduler_class='', scheduler_config='{}',
        ).items()}
        if previous:
            params['previous_loras'] = {'sourceId': previous, 'sourceKey': 'lora'}
        graph['nodes'][node_id] = {'module': 'modules.ModularDiffusers', 'action': 'Lora', 'params': params}
        previous = node_id
        order.append(node_id)
    if previous:
        graph['nodes']['models']['params']['lora_list'] = {'sourceId': previous, 'sourceKey': 'lora'}
    graph['paths'] = [order + ['models', 'generate']]
    return graph


def receipt(server, graph, *, task_id='applied-protocol', attempt=0):
    return build_resolved_execution_inputs(
        graph, server._resolved_input_records, task_id=task_id, attempt_index=attempt, node_id='generate',
    )


def projection(value):
    return next(row['appliedAdapters'] for row in value['nodes'] if row['nodeId'] == 'models')


def actual_applied_adapter_fixture(case='activated', scales=(0.0, 1.0, 0.75)):
    """Reusable client protocol fixture from the real producer and merger.

    Upstream allocation/cache selection are isolated tiny CPU unit fixtures.
    Ordinary Lora -> ModelsLoader -> consumer dispatch and PEFT activation run.
    Returned graphs/receipts contain only fixture Hub identities, no temp paths.
    This is not native model, hardware, output quality or release qualification.
    """
    if case not in {'activated', 'reused', 'empty', 'unavailable'}:
        raise ValueError('Unknown applied-adapter unit fixture case.')
    with TemporaryDirectory(prefix='modiff-applied-protocol-') as temporary, pytest.MonkeyPatch.context() as patch:
        tmp_path = Path(temporary)
        # Standalone client callers do not run tests/conftest.py. Keep registry
        # discovery isolated from the operator's installed custom extensions.
        from modiff.custom_extensions import ExtensionStore
        original = ExtensionStore.__init__
        patch.setattr(ExtensionStore, '__init__', lambda self, root=None:
                      original(self, root if root is not None else tmp_path / 'extensions'))
        runtime = native_runtime.__wrapped__(tmp_path, patch)
        fixture = graph_runtime.__wrapped__(runtime, tmp_path, patch)
        server, *_ = next(fixture)
        try:
            fixture_scales = () if case == 'empty' else scales[:1] if case == 'unavailable' else scales
            graph = hub_chain_graph(tmp_path, patch, fixture_scales,
                                    name='a' * 129 if case == 'unavailable' else None)
            execute_graph(server, graph)
            original_receipt = receipt(server, graph, task_id='original-application')
            if case == 'reused':
                execute_graph(server, graph)
            return {'receipt': receipt(server, graph), 'apiGraph': graph, 'originalReceipt': original_receipt}
        finally:
            next(fixture, None)


@pytest.mark.parametrize('scale', [0.0, 1.0, 0.75])
def test_actual_connected_graph_preserves_normalized_scale_and_exact_hub_alias(graph_runtime, tmp_path, monkeypatch, scale):
    server, *_ = graph_runtime
    graph = hub_chain_graph(tmp_path, monkeypatch, (scale,))
    execute_graph(server, graph)
    applied = projection(receipt(server, graph))
    assert applied['status'] == 'complete' and applied['observation'] == 'activated'
    item = applied['adapters'][0]
    assert item['scale'] == scale and type(item['scale']) is (int if scale in (0, 1) else float)
    assert item['artifact']['weightName'] == 'weights/adapter.safetensors'
    resolved = resolve_lora_descriptors(server.node_cache['style-0'].output['lora'])[0]
    assert resolved.weight_name == 'adapter.safetensors'
    assert resolved.artifact_weight_name == 'weights/adapter.safetensors'
    assert item['descriptorSha256'] == resolved.descriptor_sha256
    assert item['artifact']['sha256'] == resolved.content_sha256
    assert str(tmp_path) not in json.dumps(applied)


def test_connected_joint_reverse_and_valid_cache_reuse_keep_original_application_identity(graph_runtime, tmp_path, monkeypatch):
    server, *_ = graph_runtime
    graph = hub_chain_graph(tmp_path, monkeypatch)
    execute_graph(server, graph)
    first = projection(receipt(server, graph, task_id='first'))
    owner = weakref.ref(server.node_cache['models'].loader)
    assert [(r['adapterName'], r['scale']) for r in first['adapters']] == [('style_0', .75), ('style_1', 1.5)]
    execute_graph(server, graph)
    second_receipt = receipt(server, graph, task_id='second', attempt=2)
    second = projection(second_receipt)
    assert second == {**first, 'observation': 'reused'}
    assert second_receipt['taskId'] == 'second' and second_receipt['attemptIndex'] == 2
    assert server.node_cache['models'].loader is owner()
    # Retain the owned joint caches. Change only the descriptor chain order.
    graph['nodes']['style-1']['params'].pop('previous_loras')
    graph['nodes']['style-0']['params']['previous_loras'] = {'sourceId': 'style-1', 'sourceKey': 'lora'}
    graph['nodes']['models']['params']['lora_list']['sourceId'] = 'style-0'
    graph['paths'] = [['style-1', 'style-0', 'models', 'generate']]
    execute_graph(server, graph)
    reverse = projection(receipt(server, graph))
    assert reverse['observation'] == 'activated' and reverse['applicationId'] != first['applicationId']
    assert reverse['stateSha256'] != first['stateSha256']
    assert reverse['adapters'] == list(reversed(first['adapters']))


@pytest.mark.parametrize('failure', ['activation', 'scheduler', 'output', 'offload'])
def test_failed_or_partial_application_never_retains_old_success(graph_runtime, tmp_path, monkeypatch, failure):
    server, loaders, _, _, _ = graph_runtime
    graph = executable_graph([descriptor(tmp_path)])
    execute_graph(server, graph)
    old = projection(receipt(server, graph))
    graph['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, scale=0.5)]
    if failure == 'activation':
        target, name = TinyPipeline, 'set_adapters'
    else:
        target, name = loaders, {'scheduler': 'apply_lora_scheduler_override',
                                 'output': 'node_get_component_info', 'offload': 'apply_component_group_offload'}[failure]
    original = getattr(target, name)
    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        if failure != 'offload' or 'transformer' in kwargs['component_names']:
            raise RuntimeError('controlled post-mutation failure')
        return result
    monkeypatch.setattr(target, name, fail)
    with pytest.raises(RuntimeError, match='controlled post-mutation failure'):
        execute_graph(server, graph)
    node = server.node_cache['models']
    assert node._applied_adapter_state is None and node._applied_adapter_pending is None
    assert node._execution_input_record['appliedAdapters']['status'] == 'unavailable'
    del node  # The test must not keep the failed model owner alive during release.
    # Restore the real path. Recovery receives a fresh, actually activated ID.
    monkeypatch.setattr(target, name, original)
    execute_graph(server, graph)
    recovered = projection(receipt(server, graph))
    assert recovered['status'] == 'complete' and recovered['observation'] == 'activated'
    assert recovered['applicationId'] != old['applicationId']


def test_failed_final_owner_completion_clears_pending_application(graph_runtime, tmp_path, monkeypatch):
    from modiff import native_component_lora

    server, *_ = graph_runtime
    graph = executable_graph([descriptor(tmp_path)])
    original = native_component_lora.complete_component_binding
    def fail(node, state):
        original(node, state)
        raise RuntimeError('controlled final owner failure')
    monkeypatch.setattr(native_component_lora, 'complete_component_binding', fail)
    with pytest.raises(RuntimeError, match='controlled final owner failure'):
        execute_graph(server, graph)
    node = server.node_cache['models']
    assert node._applied_adapter_pending is None and node._applied_adapter_state is None
    assert node._execution_input_record['appliedAdapters']['status'] == 'unavailable'


@pytest.mark.parametrize('replacement', ['pipeline', 'output', 'output-metadata', 'owner', 'component', 'invalidated'])
def test_changed_owner_incarnation_cannot_reuse_an_old_applied_projection(graph_runtime, tmp_path, replacement):
    from modiff.applied_adapter_provenance import begin_adapter_observation, seal_adapter_observation

    server, _, manager, _, _ = graph_runtime
    graph = executable_graph([descriptor(tmp_path)])
    execute_graph(server, graph)
    node = server.node_cache['models']
    assert node._execution_input_record['appliedAdapters']['status'] == 'complete'
    if replacement == 'pipeline':
        node.loader = object()
    elif replacement == 'output':
        node.output = deepcopy(node.output)
    elif replacement == 'output-metadata':
        node.output['pipeline_components']['transformer']['model_id'] = 'foreign-owner'
    elif replacement == 'owner':
        node.node_id = 'foreign-owner'
    elif replacement == 'component':
        component_id = next(iter(manager.collections[node.node_id]))
        manager.components[component_id] = object()
    else:
        node.invalidate_cache()
    begin_adapter_observation(node, manager)
    node._applied_adapter_call_complete = True
    node._cache_reason = 'unchanged_inputs'
    seal_adapter_observation(node, manager)
    assert node._execution_input_record['appliedAdapters'] == {
        'schemaVersion': 1, 'status': 'unavailable', 'reason': 'owner-invalidated'}
    assert node._applied_adapter_state is None


def test_actual_graph_invalidation_mints_new_application_identity(graph_runtime, tmp_path):
    server, *_ = graph_runtime
    graph = executable_graph([descriptor(tmp_path)])
    execute_graph(server, graph)
    old = projection(receipt(server, graph))
    server.node_cache['models'].invalidate_cache()
    execute_graph(server, graph)
    fresh = projection(receipt(server, graph))
    assert fresh['observation'] == 'activated' and fresh['applicationId'] != old['applicationId']
    assert fresh['stateSha256'] == old['stateSha256']


def resolved_descriptor(**changes):
    return replace(ResolvedLoraDescriptor(
        descriptor_sha256='b' * 64, source='hub', repository='example/style', revision='a' * 40,
        load_directory=Path('/never-exposed'), weight_name='style.safetensors', content_sha256='c' * 64,
        adapter_name='style', scale=1.0, scheduler_class_name=None, scheduler_config={},
        artifact_weight_name='weights/style.safetensors',
    ), **changes)


@pytest.mark.parametrize('changes', [
    {'adapter_name': 'a' * 129}, {'adapter_name': 'C:/private/token'}, {'adapter_name': 'https://user:secret@host'},
    {'adapter_name': 'access_token=secret'}, {'adapter_name': '../private'},
    {'scheduler_class_name': '/private/secret'}, {'scheduler_class_name': 'https://secret'},
    {'scheduler_class_name': 'Token=secret'}, {'scheduler_class_name': 'A' * 129},
    {'repository': 'C:/private'}, {'repository': 'a' * 257}, {'revision': 'main'},
    {'artifact_weight_name': '/private.safetensors'}, {'artifact_weight_name': '../secret.safetensors'},
    {'artifact_weight_name': 'C:/secret.safetensors'}, {'artifact_weight_name': 'a' * 257},
    {'descriptor_sha256': 'B' * 64}, {'content_sha256': 'missing'}, {'scale': True},
    {'scale': float('nan')}, {'scale': 21}, {'scale': -21}, {'scale': 10**1000}, {'source': 'unknown'},
    {'scheduler_class_name': 'FlowMatchEulerDiscreteScheduler', 'scheduler_config': {'bad': object()}},
])
def test_projection_fails_closed_without_paths_secrets_or_unknown_values(changes):
    assert project_applied_adapters([resolved_descriptor(**changes)]) is None


def test_projection_has_finite_count_and_never_calls_or_stringifies_opaque_values():
    class Opaque:
        def __str__(self):
            raise AssertionError('Stringified an arbitrary object')
        def __call__(self):
            raise AssertionError('Executed an arbitrary supplier')
    assert project_applied_adapters([Opaque()]) is None
    assert project_applied_adapters(Opaque()) is None
    assert project_applied_adapters([resolved_descriptor()] * 33) is None
    assert project_applied_adapters([resolved_descriptor()] * 2) is None
    assert project_applied_adapters([resolved_descriptor(scheduler_config={'ignored': Opaque()})]) is None
    assert project_applied_adapters([])['adapters'] == []


def test_integral_scale_canonical_digest_matches_javascript_numbers():
    integer = project_applied_adapters([resolved_descriptor(scale=0), resolved_descriptor(adapter_name='second', scale=1)])
    floating = project_applied_adapters([resolved_descriptor(scale=0.0), resolved_descriptor(adapter_name='second', scale=1.0)])
    assert integer == floating
    script = "const c=require('node:crypto'); const p=JSON.parse(process.argv[1]); function s(v){return v&&typeof v==='object'?(Array.isArray(v)?'['+v.map(s).join(',')+']':'{'+Object.keys(v).sort().map(k=>JSON.stringify(k)+':'+s(v[k])).join(',')+'}'):JSON.stringify(v)};process.stdout.write(c.createHash('sha256').update(s({adapters:p.adapters})).digest('hex'))"
    result = subprocess.run(['node', '-e', script, json.dumps(integer)], check=True, capture_output=True, text=True)
    assert integer['stateSha256'] == 'sha256:applied-adapters-v1:' + result.stdout


def test_other_node_hook_keeps_v1_record_exact_and_does_not_import_models():
    code = """
import sys
from types import SimpleNamespace
from modiff.native_disk_lora import record_disk_lora_output
original = {'nodeId':'plain','module':'modules.Text','action':'String','fields':{},'omittedFields':{}}
node = SimpleNamespace(_execution_input_record=original.copy())
record_disk_lora_output(node, {})
assert node._execution_input_record == original
assert 'modules.ModularDiffusers.loaders' not in sys.modules
assert 'torch' not in sys.modules
assert 'diffusers' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('budget', ['record', 'whole-receipt'])
def test_optional_projection_preserves_existing_v1_accounting_summary_and_graph_tasks(monkeypatch, budget):
    from modiff.execution_input_provenance import capture_generation_inputs, capture_field_accounting
    from test_canonical_field_accounting import ddim_graph, run_ddim

    original, *_ = run_ddim(monkeypatch)
    graph = ddim_graph()
    records = {row['nodeId']: deepcopy(row) for row in original['nodes']}
    # Serializer-only stress records: retain the real DDIM task graph and an
    # ordinary Preview ancestry. This does not grant Preview producer authority.
    owner = {'module': 'modules.Image', 'action': 'Preview',
             'params': {'image': {'sourceId': 'node-05', 'sourceKey': 'output'}}}
    graph['nodes']['budget-owner'] = owner
    record = capture_generation_inputs('budget-owner', owner, {'seed': 1} if budget == 'whole-receipt' else
                                      {'prompt': 'p' * 16000, 'prompt_2': 'q' * 16000, 'prompt_3': 'r' * 16000})
    records['budget-owner'] = record
    previous = 'budget-owner'
    if budget == 'whole-receipt':
        for index in range(15):
            node_id = f'preview-budget-{index}'
            definition = {'module': 'modules.Image', 'action': 'Preview', 'params': {
                'image': {'sourceId': previous, 'sourceKey': 'output'}, 'export': {'value': '0' * 8000}}}
            graph['nodes'][node_id] = definition
            row = capture_generation_inputs(node_id, definition, {'width': 32})
            row['fieldAccounting'] = capture_field_accounting(definition, {'export': '0' * 8000},
                                                             consumed={'export': '0' * 8000})
            records[node_id] = bind_generation_input_origins(node_id, definition, row)
            previous = node_id
    kwargs = dict(task_id='budget-only', attempt_index=0, node_id=previous)
    old = build_resolved_execution_inputs(graph, records, **kwargs)
    large = project_applied_adapters([resolved_descriptor(
        adapter_name=f'style_{index}', repository='example/' + 'r' * 80,
        artifact_weight_name='w' * 240 + '.safetensors',
    ) for index in range(32)])
    assert large is not None
    record['appliedAdapters'] = {'schemaVersion': 1, 'status': 'complete',
                                'source': 'backend-adapter-application', 'applicationId': 'a' * 32,
                                'observation': 'activated', **large}
    new = build_resolved_execution_inputs(graph, records, **kwargs)
    assert new['appliedAdaptersStatus'] == {'schemaVersion': 1, 'truncated': True}
    comparable = {key: value for key, value in new.items() if key != 'appliedAdaptersStatus'}
    comparable['nodes'] = [{key: value for key, value in row.items() if key != 'appliedAdapters'} for row in new['nodes']]
    assert comparable == old
    assert not old['truncated'] and not new['truncated']
    assert old.get('graphTasks') and new['summary']
    assert len(json.dumps(new['nodes'], ensure_ascii=True).encode('ascii')) < 262_144
    if budget == 'record':
        applied = next(row['appliedAdapters'] for row in new['nodes'] if row['nodeId'] == 'budget-owner')
        assert applied == {'schemaVersion': 1, 'status': 'truncated', 'reason': 'record-size-limit'}


@pytest.mark.parametrize('case', ['activated', 'reused', 'empty', 'unavailable'])
def test_reusable_actual_producer_fixture_has_no_private_paths(case):
    value = actual_applied_adapter_fixture(case)
    applied = projection(value['receipt'])
    assert 'modiff-applied-protocol-' not in json.dumps(value)
    assert value['receipt']['appliedAdaptersStatus'] == {'schemaVersion': 1, 'truncated': False}
    if case == 'unavailable':
        assert applied == {'schemaVersion': 1, 'status': 'unavailable', 'reason': 'unsupported-projection'}
    elif case == 'empty':
        assert applied['status'] == 'complete' and applied['adapters'] == []
    else:
        assert [row['scale'] for row in applied['adapters']] == [0, 1, .75]
        assert applied['observation'] == case
        if case == 'reused':
            assert applied['applicationId'] == projection(value['originalReceipt'])['applicationId']
