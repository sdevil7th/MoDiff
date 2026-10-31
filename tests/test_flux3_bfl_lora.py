"""Real single-file tiny BFL components through selected app LoRA ownership.

Only synthetic local artifact declarations replace pretrained acquisition. The
official ComponentSpec, converter, quantizer, PEFT and normal graph dispatcher
execute. Stage/media output, CUDA kernels and trained-model fit are separate.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from test_flux3_native_lora import PAIRS, TARGET, descriptor, prediction, source_graph
import test_modular_weight_formats as weight_format_tests

artifacts = weight_format_tests.runtime


@pytest.fixture
def runtime(artifacts, monkeypatch):
    import modules
    from modiff.server import WebServer, memory_manager

    monkeypatch.setattr(memory_manager, 'cache', {})
    monkeypatch.delenv('MODIFF_SUPERVISOR_QUEUE_STATE', raising=False)
    server = WebServer(modules=modules.MODULE_MAP)
    monkeypatch.setattr(server, 'queue_message', lambda _value: None)
    monkeypatch.setattr(server, '_runtime_fingerprint', lambda **_kwargs: {'fingerprint': 'tiny-bfl-cpu-unit-only'})
    monkeypatch.setattr(server, '_gpu_process_snapshot', lambda: {'observation': 'no-GPU-unit'})
    monkeypatch.setattr(server, '_record_auto_resource_success', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, '_record_optimization_observations', lambda *_args, **_kwargs: [])
    actual = server.execute_node

    class ReachedFirstStage(Exception):
        pass

    def dispatch(identity, node, sid, **kwargs):
        if node['action'] not in ('ModelsLoader', 'Lora'):
            raise ReachedFirstStage('Real loader/lifecycle proof stops before stage/media execution.')
        try:
            return actual(identity, node, sid, quiet=True)
        except Exception:
            loader_node = server.node_cache.get(identity)
            pipeline = getattr(loader_node, 'loader', None)
            model = getattr(pipeline, 'transformer', None)
            if model is not None:
                spec = pipeline.get_component_spec('transformer')
                mapping = getattr(model, 'hf_device_map', None)
                print('ACTUAL_BFL_FAILURE_IDENTITY', json.dumps({
                    'componentType': type(model).__name__, 'specType': spec.type_hint.__name__,
                    'revision': spec.revision, 'loadIdMatches': getattr(model, '_diffusers_load_id', None) == spec.load_id,
                    'deviceMap': {key: str(value) for key, value in mapping.items()} if isinstance(mapping, dict) else None,
                    'deviceMapTypes': {key: type(value).__name__ for key, value in mapping.items()} if isinstance(mapping, dict) else None,
                    'hookPresent': hasattr(model, '_diffusers_hook'),
                    'hookValueType': type(getattr(model, '_diffusers_hook', None)).__name__,
                    'parameterDevices': sorted({str(value.device) for value in model.parameters()}),
                    'bufferDevices': sorted({str(value.device) for value in model.buffers()}),
                }), flush=True)
            raise

    monkeypatch.setattr(server, 'execute_node', dispatch)

    def run(graph):
        server.current_task = {'task_id': 'bfl-cpu-unit', 'sid': graph['sid'], 'status': 'running', 'progress': 0}
        with pytest.raises(ReachedFirstStage):
            server._execute_graph(deepcopy(graph))
        return server.node_cache['models']

    yield SimpleNamespace(server=server, manager=artifacts.manager, run=run, artifacts=artifacts)
    server.node_cache.clear()
    memory_manager.cache.clear()
    artifacts.manager.components.clear()
    artifacts.manager.collections.clear()
    gc.collect()


def graph(pair, scheme, items=None, *, task='text_to_image', chain=False):
    value = source_graph(pair, items, task=task, chain=chain)
    value['nodes']['models']['params']['weight_format'] = {'value': scheme}
    return value


def packed_state(model):
    from peft.tuners.tuners_utils import BaseTunerLayer
    layer = model.get_submodule(TARGET)
    if isinstance(layer, BaseTunerLayer):
        layer = layer.get_base_layer()
    assert layer.in_features == layer.out_features == 32
    return {name: value.detach().reshape(-1).view(torch.uint8).clone()
            for name, value in [('weight', layer.weight), *layer.named_buffers()]}


def assert_packed_state(model, expected):
    actual = packed_state(model)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in expected)


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
@pytest.mark.parametrize('task', ['text_to_image', 'edit_image'])
def test_real_selected_bfl_lora_loader_dispatch(runtime, tmp_path, pair, scheme, task):
    baseline_node = runtime.run(graph(pair, scheme, task=task))
    baseline = prediction(runtime)
    packed = packed_state(baseline_node.loader.transformer)
    former = weakref.ref(baseline_node.loader.transformer)
    del baseline_node
    node = runtime.run(graph(pair, scheme, [descriptor(tmp_path, original=True)], task=task))
    assert former() is None
    assert node._native_disk_lora_state['componentComplete'] is True
    assert node.loader.transformer.active_adapters() == ['one']
    assert node.loader.vae.dtype == torch.float32
    assert not getattr(node.loader.text_encoder, 'peft_config', {})
    changed = prediction(runtime)
    assert torch.isfinite(changed).all() and not torch.allclose(changed, baseline)
    assert_packed_state(node.loader.transformer, packed)


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
@pytest.mark.parametrize('task', ['text_to_image', 'edit_image'])
@pytest.mark.parametrize('submission', ['node', 'literal'])
def test_real_selected_single_bfl_lora_loader_dispatch(runtime, tmp_path, pair, scheme, task, submission):
    baseline_node = runtime.run(graph(pair, scheme, task=task))
    baseline = prediction(runtime)
    packed = packed_state(baseline_node.loader.transformer)
    former = weakref.ref(baseline_node.loader.transformer)
    del baseline_node
    item = descriptor(tmp_path, original=True)
    source = graph(pair, scheme, [item], task=task, chain=submission == 'node')
    if submission == 'literal':
        source['nodes']['models']['params']['lora_list']['value'] = item
    node = runtime.run(source)
    if submission == 'node':
        assert type(runtime.server.node_cache['adapter-0'].output['lora']) is dict
    assert former() is None
    assert node._native_disk_lora_state['componentComplete'] is True
    assert node.loader.transformer.active_adapters() == ['one']
    changed = prediction(runtime)
    assert torch.isfinite(changed).all() and not torch.allclose(changed, baseline)
    assert_packed_state(node.loader.transformer, packed)


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
def test_bfl_order_zero_reenable_removal_replace_and_warm_owner(runtime, tmp_path, pair, scheme):
    runtime.run(graph(pair, scheme))
    baseline = prediction(runtime)
    packed = packed_state(runtime.server.node_cache['models'].loader.transformer)
    one = descriptor(tmp_path)
    node = runtime.run(graph(pair, scheme, [one]))
    changed = prediction(runtime)
    previous = weakref.ref(node.loader)
    assert runtime.run(graph(pair, scheme, [one])).loader is previous()
    del node
    two = descriptor(tmp_path, name='two', scale=.35, original=True)
    node = runtime.run(graph(pair, scheme, [two, one], chain=True))
    assert previous() is None
    assert node.loader.transformer.active_adapters() == ['two', 'one']
    both = prediction(runtime)
    assert torch.isfinite(both).all() and not torch.allclose(both, changed)
    assert_packed_state(node.loader.transformer, packed)
    del node
    runtime.run(graph(pair, scheme, [one, two], chain=True))
    assert runtime.server.node_cache['models'].loader.transformer.active_adapters() == ['one', 'two']
    runtime.run(graph(pair, scheme, [descriptor(tmp_path, scale=0.)]))
    torch.testing.assert_close(prediction(runtime), baseline, rtol=0, atol=0)
    runtime.run(graph(pair, scheme, [one]))
    torch.testing.assert_close(prediction(runtime), changed, rtol=0, atol=0)
    replacement = descriptor(tmp_path, seed=109)
    runtime.run(graph(pair, scheme, [replacement]))
    assert not torch.allclose(prediction(runtime), changed)
    runtime.run(graph(pair, scheme, []))
    model = runtime.server.node_cache['models'].loader.transformer
    assert not getattr(model, 'peft_config', {})
    assert_packed_state(model, packed)
    torch.testing.assert_close(prediction(runtime), baseline, rtol=0, atol=0)


@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
@pytest.mark.parametrize('bad', ['shape', 'nan', 'metadata_dora', 'metadata_config'])
@pytest.mark.parametrize('submission', ['list', 'node', 'literal'])
def test_bfl_bad_factors_preserve_working_owner(runtime, tmp_path, scheme, bad, submission):
    node = runtime.run(graph(PAIRS[0], scheme, [descriptor(tmp_path)]))
    previous = weakref.ref(node.loader)
    del node
    item = descriptor(tmp_path, bad=bad)
    source = graph(PAIRS[0], scheme, [item], chain=submission == 'node')
    if submission == 'literal':
        source['nodes']['models']['params']['lora_list']['value'] = item
    with pytest.raises(ValueError):
        runtime.run(source)
    assert runtime.server.node_cache['models'].loader is previous()


@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
def test_bfl_single_descriptor_without_source_context_preserves_working_owner(runtime, tmp_path, monkeypatch, scheme):
    from diffusers import ComponentSpec
    item = descriptor(tmp_path)
    node = runtime.run(graph(PAIRS[0], scheme, [item]))
    previous = weakref.ref(node.loader)
    expected = prediction(runtime)
    monkeypatch.setattr(ComponentSpec, 'load', lambda *_args, **_kwargs: pytest.fail('Unselected dispatch must not allocate.'))
    with pytest.raises(ValueError, match='source-selected|source authority'):
        runtime.artifacts.call(PAIRS[0], scheme, node=node, lora_list=item)
    assert runtime.server.node_cache['models'].loader is previous()
    torch.testing.assert_close(prediction(runtime), expected, rtol=0, atol=0)


@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
def test_bfl_post_injection_failure_poisons_owner_and_clean_retry(runtime, tmp_path, monkeypatch, scheme):
    from diffusers import Flux3ModularPipeline
    original = Flux3ModularPipeline.load_lora_weights
    observed = []

    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        observed.append(weakref.ref(self.transformer))
        assert getattr(self.transformer, 'peft_config', {})
        raise RuntimeError('Injected after real BFL PEFT mutation.')

    monkeypatch.setattr(Flux3ModularPipeline, 'load_lora_weights', fail)
    source = graph(PAIRS[0], scheme, [descriptor(tmp_path)])
    with pytest.raises(RuntimeError, match='real BFL PEFT'):
        runtime.run(source)
    node = runtime.server.node_cache['models']
    assert node._native_disk_lora_state['componentComplete'] is False
    assert not node._cache_valid
    monkeypatch.setattr(Flux3ModularPipeline, 'load_lora_weights', original)
    del node
    runtime.run(source)
    assert all(reference() is None for reference in observed)
    assert runtime.server.node_cache['models']._native_disk_lora_state['componentComplete'] is True


@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
def test_bfl_actual_hook_before_peft_refuses(runtime, tmp_path, monkeypatch, scheme):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from diffusers import ComponentSpec
    original = ComponentSpec.load
    mutated = []

    def hook(self, **kwargs):
        value = original(self, **kwargs)
        if self.name == 'transformer':
            add_hook_to_module(value.get_submodule(TARGET), AlignDevicesHook(execution_device='cpu', io_same_device=True))
            mutated.append(value)
        return value

    monkeypatch.setattr(ComponentSpec, 'load', hook)
    with pytest.raises(RuntimeError, match='Hooked'):
        runtime.run(graph(PAIRS[0], scheme, [descriptor(tmp_path)]))
    assert mutated and not getattr(mutated[0], 'peft_config', {})


def test_bfl_context_free_updater_cannot_borrow_loaded_quantized_owner(runtime, tmp_path):
    from modules.ModularDiffusers import loaders
    node = runtime.artifacts.call()
    with pytest.raises(ValueError, match='source-selected'):
        loaders.update_lora_adapters(node.loader, [descriptor(tmp_path)])
    assert not getattr(node.loader.transformer, 'peft_config', {})


@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
def test_bfl_bf16_compute_and_actual_adapter_dtype(runtime, tmp_path, scheme):
    source = graph(PAIRS[0], scheme)
    source['nodes']['models']['params']['dtype']['value'] = 'bfloat16'
    source['nodes']['models']['params']['device']['value'] = 'cpu:0'
    runtime.run(source)
    generator = torch.Generator().manual_seed(59)
    values = dict(hidden_states=torch.randn(1, 8, 4, generator=generator, dtype=torch.bfloat16),
                  encoder_hidden_states=torch.randn(1, 8, 32, generator=generator, dtype=torch.bfloat16),
                  timestep=torch.tensor([.5]), img_ids=torch.zeros(8, 4), txt_ids=torch.zeros(8, 4))
    baseline = runtime.server.node_cache['models'].loader.transformer(**values).sample.detach().clone()
    source['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path)]
    node = runtime.run(source)
    model = node.loader.transformer
    wrapped = model.get_submodule(TARGET)
    assert model.hf_quantizer.compute_dtype == torch.bfloat16
    assert wrapped.get_base_layer().compute_dtype == torch.bfloat16
    assert wrapped.lora_A['one'].weight.dtype == torch.bfloat16
    assert wrapped.lora_B['one'].weight.dtype == torch.bfloat16
    changed = model(**values).sample
    assert changed.dtype == torch.bfloat16 and torch.isfinite(changed).all()
    assert not torch.equal(changed, baseline)


@pytest.mark.parametrize('field', ['weight', 'config'])
def test_bfl_artifact_drift_refuses_before_working_owner_release(runtime, tmp_path, field):
    source = graph(PAIRS[0], 'fp8r', [descriptor(tmp_path)])
    node = runtime.run(source)
    former = weakref.ref(node.loader)
    expected = prediction(runtime)
    del node
    row = next(row for row in runtime.artifacts.recipes if row['modelType'] == PAIRS[0][0] and row['scheme'] == 'fp8r')
    target = runtime.artifacts.snapshots[PAIRS[0][0]] / row[field]['path']
    raw = target.read_bytes()
    target.rename(target.with_name(target.name + '.mapped-old'))
    target.write_bytes(raw + b' ')
    source['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, scale=.35)]
    with pytest.raises(ValueError, match='size/SHA-256'):
        runtime.run(source)
    assert runtime.server.node_cache['models'].loader is former()
    torch.testing.assert_close(prediction(runtime), expected, rtol=0, atol=0)


@pytest.mark.parametrize('fault', ['shared', 'external'])
def test_bfl_borrowed_target_stops_before_replacement_allocation(runtime, tmp_path, monkeypatch, fault):
    from diffusers import ComponentSpec
    source = graph(PAIRS[0], 'fp8r', [descriptor(tmp_path)])
    node = runtime.run(source)
    retained = node.loader.transformer
    reference = weakref.ref(retained)
    if fault == 'shared':
        runtime.manager.add('transformer', retained, collection='foreign-owner')
    del node
    loads = []
    original = ComponentSpec.load

    def observe(self, **kwargs):
        loads.append(self.name)
        return original(self, **kwargs)

    monkeypatch.setattr(ComponentSpec, 'load', observe)
    source['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, scale=.35)]
    with pytest.raises(ValueError, match='shared' if fault == 'shared' else 'referenced|references'):
        runtime.run(source)
    assert not loads and reference() is retained
    del retained
    if fault == 'external':
        gc.collect()
        runtime.run(source)
        assert reference() is None and 'transformer' in loads


@pytest.mark.parametrize('mapping', [{'': torch.device('cpu'), 'foreign': torch.device('meta')}, {'': torch.device('meta')}])
def test_bfl_distributed_or_foreign_load_map_refuses_before_peft(runtime, tmp_path, monkeypatch, mapping):
    from diffusers import ComponentSpec
    original = ComponentSpec.load
    observed = []

    def change(self, **kwargs):
        value = original(self, **kwargs)
        if self.name == 'transformer':
            value.hf_device_map = mapping
            observed.append(value)
        return value

    monkeypatch.setattr(ComponentSpec, 'load', change)
    with pytest.raises(RuntimeError, match='single-root CPU load map'):
        runtime.run(graph(PAIRS[0], 'fp8r', [descriptor(tmp_path)]))
    assert observed and not getattr(observed[0], 'peft_config', {})


def test_bfl_peft_internal_path_cannot_destroy_working_owner(runtime, tmp_path):
    from safetensors.torch import load_file, save_file
    from modiff.auxiliary_lora import build_lora_descriptor
    source = graph(PAIRS[0], 'fp8r', [descriptor(tmp_path)])
    node = runtime.run(source)
    previous = weakref.ref(node.loader)
    del node
    item = descriptor(tmp_path, name='nested')
    artifact = item['artifact']
    path = Path(artifact['root']) / artifact['weight_name']
    state = load_file(path)
    save_file({key.replace(TARGET + '.', TARGET + '.base_layer.'): value for key, value in state.items()}, path)
    item = build_lora_descriptor(selection={'source': 'local', 'value': artifact['root']},
        weight_name=artifact['weight_name'], adapter_name='nested', scale=1., revision=None,
        expected_sha256=None, scheduler_class='', scheduler_config=None)
    source['nodes']['models']['params']['lora_list']['value'] = [item]
    with pytest.raises(ValueError, match='actual target shape'):
        runtime.run(source)
    assert runtime.server.node_cache['models'].loader is previous()
