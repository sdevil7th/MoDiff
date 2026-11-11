"""Selected native CPU LoRA lifecycle; tiny acquisition is an explicit fixture.

Actual source graph, ModelsLoader, NodeBase, component manager, immutable local
adapters and official PEFT mutation execute. No pretrained files or hardware fit
are established. Stage/media execution is covered separately from this loader
and owner-lifetime test; dispatch deliberately stops at the first stage.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
from safetensors.torch import save_file

from modiff.auxiliary_lora import build_lora_descriptor


PAIRS = [
    ('Flux3ModularPipeline', 'black-forest-labs/flux-3-image-klein-base', '1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4'),
    ('Flux3DistilledModularPipeline', 'black-forest-labs/flux-3-image-klein', '8454017f9ebaada30c998fccd9f67a80c7580f68'),
]
TARGET = 'mode_blocks.image.0.attn.to_q'


def require_flux3_api():
    import diffusers
    names = ('Flux3Transformer2DModel', 'Flux3ModularPipeline', 'Flux3DistilledModularPipeline')
    missing = [name for name in names if getattr(diffusers, name, None) is None]
    if missing:
        pytest.skip('Official Flux3 API unavailable in this runtime: ' + ', '.join(missing))


def tiny_transformer(model):
    from diffusers import Flux3Transformer2DModel
    torch.manual_seed(17)
    return Flux3Transformer2DModel(
        in_channels=4, cond_in_channels=4, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
        timestep_channels=32, axes_dims_rope=(4, 4, 4, 4),
        flow_map='Distilled' in model,
    ).eval()


def descriptor(tmp_path, name='one', scale=1.0, *, original=False, bad=None, seed=None):
    directory = tmp_path / (name + ('-original' if original else '-ordinary') + ('-' + str(bad) if bad else ''))
    directory.mkdir(exist_ok=True)
    prefix = 'diffusion_model.content_mode_blocks.image.0.q_proj' if original else 'transformer.' + TARGET
    generator = torch.Generator().manual_seed(seed if seed is not None else 23 if name == 'one' else 29)
    state = {prefix + '.lora_A.weight': torch.randn(2, 32, generator=generator) * .2,
             prefix + '.lora_B.weight': torch.randn(32, 2, generator=generator) * .1}
    if bad == 'shape':
        state[prefix + '.lora_A.weight'] = torch.ones(2, 31)
    elif bad == 'rank':
        state[prefix + '.lora_B.weight'] = torch.ones(32, 3)
    elif bad == 'nan':
        state[prefix + '.lora_A.weight'][0, 0] = float('nan')
    elif bad == 'infinity':
        state[prefix + '.lora_B.weight'][0, 0] = float('inf')
    elif bad == 'dora':
        state[prefix + '.dora_scale'] = torch.ones(32)
    elif bad == 'foreign':
        state = {key.replace('transformer.', 'text_encoder.'): value for key, value in state.items()}
    elif bad == 'mixed':
        state['diffusion_model.content_mode_blocks.image.0.q_proj.lora_B.weight'] = state.pop(prefix + '.lora_B.weight')
    metadata = None
    if bad in ('metadata_dora', 'metadata_config'):
        metadata = {'lora_adapter_metadata': json.dumps({
            'transformer.r': 2, 'transformer.lora_alpha': 2, 'transformer.target_modules': [TARGET],
            'transformer.use_dora' if bad == 'metadata_dora' else 'transformer.use_rslora': True,
        })}
    elif bad == 'inert_metadata':
        metadata = {'format': 'pt', 'note': 'Inert Safetensors metadata is preserved.'}
    file = directory / 'adapter.safetensors'
    save_file(state, file, metadata=metadata)
    return build_lora_descriptor(selection={'source': 'local', 'value': str(directory)}, weight_name=file.name,
                                 adapter_name=name, scale=scale, revision=None, expected_sha256=None,
                                 scheduler_class='', scheduler_config=None)


def source_graph(pair, items=None, *, task='text_to_image', chain=False):
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    model, repo, revision = pair
    spec = studio_execution_spec_for_pair(model, task)
    values = dict(model_type=model, repo_id={'source': 'hub', 'value': repo}, revision=revision,
                  workflow_id='text2image' if task == 'text_to_image' else 'image_conditioned', reviewed_variant=None,
                  dtype='float32', device='cpu', offload_mode='none', auto_offload=False,
                  weight_loading_mode='cpu', lora_list=items, unet=None, vae=None, controlnet=None,
                  quant_config=None, trust_remote_code=False)
    nodes = {'models': {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader',
                       'params': {key: {'value': value} for key, value in values.items()}}}
    for role, key, *_ in spec['roles']:
        if role == 'models':
            continue
        module, action = key.rsplit('.', 1)
        params = {}
        if role in ('guidance', 'guider'):
            params = {'model_type': {'value': model}, 'guider': {'value': 'classifierFreeGuidance'},
                      'guidance_scale': {'value': 3.5}}
        nodes[role] = {'module': module, 'action': action, 'params': params}
    for source, output, target, field in spec['edges']:
        nodes[target]['params'][field] = {'sourceId': source, 'sourceKey': output}
    path = list(nodes)
    if chain:
        previous = None
        for index, item in enumerate(items):
            identity = 'adapter-' + str(index)
            artifact = item['artifact']
            values = dict(model={'source': 'local', 'value': artifact['root']}, weight_name=artifact['weight_name'],
                          expected_sha256=artifact['sha256'], adapter_name=item['adapter_name'], scale=item['scale'],
                          revision='', scheduler_class='', scheduler_config='{}')
            params = {key: {'value': value} for key, value in values.items()}
            params['previous_loras'] = {'value': None} if previous is None else {'sourceId': previous, 'sourceKey': 'lora'}
            nodes[identity] = {'module': 'modules.ModularDiffusers', 'action': 'Lora', 'params': params}
            path.insert(index, identity)
            previous = identity
        nodes['models']['params']['lora_list'] = {'sourceId': previous, 'sourceKey': 'lora'}
    return {'sid': 'flux3-unit', 'nodes': nodes, 'paths': [path]}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    require_flux3_api()
    from diffusers import AutoencoderKLFlux2, ComponentSpec, ComponentsManager
    from transformers import (Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor)
    from modiff.config import CONFIG
    import modules
    import modules.ModularDiffusers as modular
    from modules.ModularDiffusers import loaders
    from modiff.server import WebServer, memory_manager

    for field in ('work_dir', 'data', 'images', 'videos', 'audio', 'models', 'upscalers', 'temp'):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(tmp_path / 'hub'))
    monkeypatch.setitem(CONFIG.hf, 'online_status', 'Offline')
    monkeypatch.delenv('MODIFF_SUPERVISOR_QUEUE_STATE', raising=False)
    manager, events, faults = ComponentsManager(), [], {}
    monkeypatch.setattr(modular, 'components', manager)
    monkeypatch.setattr(loaders, 'components', manager)
    monkeypatch.setattr(memory_manager, 'cache', {})
    for _model, repo, revision in PAIRS:
        snapshot = Path(CONFIG.hf['cache_dir']) / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
        snapshot.mkdir(parents=True)
        fixture = Path(__file__).parent / 'fixtures' / (repo.rsplit('/', 1)[1] + '.modular-index.v1.json')
        (snapshot / 'modular_model_index.json').write_bytes(fixture.read_bytes())

    def construct(spec, **kwargs):
        pair = next(pair for pair in PAIRS if pair[1] == spec.pretrained_model_name_or_path)
        assert spec.revision == pair[2] and kwargs.get('trust_remote_code') is False
        events.append((spec.name, pair[0]))
        if spec.name == 'transformer':
            value = tiny_transformer(pair[0])
            if faults.get('hook'):
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module
                add_hook_to_module(value, AlignDevicesHook(execution_device='cpu', io_same_device=True))
            if faults.get('packed'):
                from diffusers.quantizers.bfl.utils import BFLFP8RLinear, quantize_fp8r
                original = value.get_submodule(TARGET)
                packed, scales = quantize_fp8r(original.weight)
                layer = BFLFP8RLinear(32, 32, bias=False, compute_dtype=torch.float32)
                layer.weight.data.copy_(packed)
                layer.weight_scale.copy_(scales)
                value.mode_blocks['image'][0].attn.to_q = layer
        elif spec.name == 'vae':
            value = AutoencoderKLFlux2(block_out_channels=(4, 4, 4, 4),
                down_block_types=('DownEncoderBlock2D',) * 4, up_block_types=('UpDecoderBlock2D',) * 4,
                layers_per_block=1, latent_channels=2, norm_num_groups=1, mid_block_add_attention=False).eval()
        elif spec.name == 'text_encoder':
            value = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
                text_config={'vocab_size': 16, 'hidden_size': 8, 'intermediate_size': 16,
                             'num_hidden_layers': 1, 'num_attention_heads': 1, 'num_key_value_heads': 1,
                             'head_dim': 8, 'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2]}},
                vision_config={'depth': 1, 'hidden_size': 8, 'intermediate_size': 16, 'num_heads': 1,
                               'out_hidden_size': 8, 'patch_size': 2, 'spatial_merge_size': 1, 'temporal_patch_size': 1},
                image_token_id=12, video_token_id=13, vision_start_token_id=14, vision_end_token_id=15)).eval()
        elif spec.name == 'tokenizer':
            tokenizer = Qwen2TokenizerFast(vocab={'<|endoftext|>': 0, '<|vision_start|>': 1, 'a': 2}, merges=[])
            value = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer,
                                     video_processor=Qwen3VLVideoProcessor())
        else:
            raise AssertionError('Unexpected pretrained component: ' + spec.name)
        assert type(value) is spec.type_hint
        value._diffusers_load_id = spec.load_id
        return value

    monkeypatch.setattr(ComponentSpec, 'load', construct)
    server = WebServer(modules=modules.MODULE_MAP, work_dir=str(tmp_path), data_dir=str(tmp_path / 'data'))
    monkeypatch.setattr(server, 'queue_message', lambda _value: None)
    monkeypatch.setattr(server, '_runtime_fingerprint', lambda **_kwargs: {'fingerprint': 'tiny-cpu-unit-only'})
    monkeypatch.setattr(server, '_gpu_process_snapshot', lambda: {'observation': 'no-GPU-unit'})
    monkeypatch.setattr(server, '_record_auto_resource_success', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, '_record_optimization_observations', lambda *_args, **_kwargs: [])
    actual = server.execute_node

    class ReachedFirstStage(Exception):
        pass

    def dispatch(identity, node, sid, **kwargs):
        if node['action'] not in ('ModelsLoader', 'Lora'):
            raise ReachedFirstStage('Loader/lifecycle proof stops before stage and media execution.')
        return actual(identity, node, sid, quiet=True)

    monkeypatch.setattr(server, 'execute_node', dispatch)

    def run(graph):
        server.current_task = {'task_id': 'cpu-unit', 'sid': graph['sid'], 'status': 'running', 'progress': 0}
        with pytest.raises(ReachedFirstStage):
            server._execute_graph(deepcopy(graph))
        return server.node_cache['models']

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        yield SimpleNamespace(server=server, manager=manager, events=events, run=run, loaders=loaders, faults=faults)
    server.node_cache.clear()
    memory_manager.cache.clear()
    manager.components.clear()
    manager.collections.clear()
    gc.collect()
    torch.set_num_threads(threads)


def prediction(runtime):
    model = runtime.server.node_cache['models'].loader.transformer
    generator = torch.Generator().manual_seed(31)
    values = dict(hidden_states=torch.randn(1, 8, 4, generator=generator),
                  encoder_hidden_states=torch.randn(1, 8, 32, generator=generator),
                  timestep=torch.tensor([.5]), img_ids=torch.zeros(8, 4), txt_ids=torch.zeros(8, 4))
    if model.config.flow_map:
        values['target_timestep'] = torch.tensor([.25])
    return model(**values).sample.detach().clone()


def test_context_free_flux3_updater_refuses_before_official_injection(tmp_path):
    require_flux3_api()
    from diffusers import Flux3ModularPipeline
    from modules.ModularDiffusers import loaders
    pipeline = Flux3ModularPipeline(workflow='text2image')
    pipeline.register_components(transformer=tiny_transformer(PAIRS[0][0]))
    with pytest.raises(ValueError, match='source-selected'):
        loaders.update_lora_adapters(pipeline, [descriptor(tmp_path)])
    assert not getattr(pipeline.transformer, 'peft_config', {})


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize('task', ['text_to_image', 'edit_image'])
def test_real_native_loader_issues_transformer_only_complete_binding(runtime, tmp_path, pair, task):
    node = runtime.run(source_graph(pair, [descriptor(tmp_path, original=True)], task=task))
    assert node._native_disk_lora_state['componentComplete'] is True
    assert node._native_disk_lora_state['componentChecker'] is None
    assert node.loader.transformer.active_adapters() == ['one']
    assert not getattr(node.loader.text_encoder, 'peft_config', {})
    assert torch.isfinite(prediction(runtime)).all()


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
def test_actual_order_scale_zero_remove_replace_and_owner_cache(runtime, tmp_path, pair):
    source = source_graph(pair)
    runtime.run(source)
    baseline = prediction(runtime)
    item = descriptor(tmp_path)
    source['nodes']['models']['params']['lora_list']['value'] = [item]
    runtime.run(source)
    changed = prediction(runtime)
    assert not torch.allclose(changed, baseline)
    previous = weakref.ref(runtime.server.node_cache['models'].loader)
    count = len(runtime.events)
    runtime.run(source)
    assert runtime.server.node_cache['models'].loader is previous() and len(runtime.events) == count
    second = descriptor(tmp_path, name='two', scale=.35, original=True)
    source = source_graph(pair, [second, item], chain=True)
    runtime.run(source)
    assert previous() is None
    assert runtime.server.node_cache['models'].loader.transformer.active_adapters() == ['two', 'one']
    assert not torch.allclose(prediction(runtime), changed)
    source = source_graph(pair, [item, second], chain=True)
    runtime.run(source)
    assert runtime.server.node_cache['models'].loader.transformer.active_adapters() == ['one', 'two']
    zero = descriptor(tmp_path, scale=0.)
    source = source_graph(pair, [zero])
    runtime.run(source)
    torch.testing.assert_close(prediction(runtime), baseline, rtol=0, atol=0)
    source['nodes']['models']['params']['lora_list']['value'] = [item]
    runtime.run(source)
    torch.testing.assert_close(prediction(runtime), changed, rtol=0, atol=0)
    source['nodes']['models']['params']['lora_list']['value'] = []
    runtime.run(source)
    assert not getattr(runtime.server.node_cache['models'].loader.transformer, 'peft_config', {})
    torch.testing.assert_close(prediction(runtime), baseline, rtol=0, atol=0)


@pytest.mark.parametrize('bad', ['rank', 'nan', 'infinity', 'dora', 'foreign', 'mixed', 'metadata_dora', 'metadata_config'])
def test_bad_factor_payload_refuses_before_new_component_acquisition(runtime, tmp_path, bad):
    with pytest.raises(ValueError):
        runtime.run(source_graph(PAIRS[0], [descriptor(tmp_path, bad=bad)]))
    assert not runtime.events


def test_inert_safetensors_metadata_retains_the_ordinary_selected_route(runtime, tmp_path):
    node = runtime.run(source_graph(PAIRS[0], [descriptor(tmp_path, bad='inert_metadata')]))
    assert node.loader.transformer.active_adapters() == ['one']
    assert node._native_disk_lora_state['componentComplete'] is True


def test_retained_actual_shape_refuses_before_destroying_working_owner(runtime, tmp_path):
    source = source_graph(PAIRS[0], [descriptor(tmp_path)])
    runtime.run(source)
    previous = weakref.ref(runtime.server.node_cache['models'].loader)
    count = len(runtime.events)
    source['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, bad='shape')]
    with pytest.raises(ValueError, match='actual target shape'):
        runtime.run(source)
    assert previous() is runtime.server.node_cache['models'].loader and len(runtime.events) == count


@pytest.mark.parametrize('fault', ['placement', 'quantization', 'weight_format', 'foreign_owner', 'material', 'duplicate', 'task', 'revision'])
def test_source_scope_refuses_before_component_construction(runtime, tmp_path, fault):
    source = source_graph(PAIRS[0], [descriptor(tmp_path)])
    params = source['nodes']['models']['params']
    if fault == 'placement':
        params['offload_mode']['value'] = 'group_disk'
    elif fault == 'quantization':
        params['quant_config']['value'] = {'transformer': {'quant_method': 'bfl'}}
    elif fault == 'weight_format':
        params['weight_format'] = {'value': 'fp8r'}
    elif fault == 'foreign_owner':
        source['nodes']['denoise']['params']['unet']['sourceId'] = 'foreign'
    elif fault == 'material':
        source['nodes']['denoise']['params']['embeddings']['sourceKey'] = 'wrong'
    elif fault == 'duplicate':
        source['nodes']['duplicate'] = deepcopy(source['nodes']['prompt'])
        source['paths'][0].append('duplicate')
    elif fault == 'task':
        params['workflow_id']['value'] = 'img2img'
    else:
        params['revision']['value'] = 'a' * 40
    # FP8r now has an exact source contract, but this native-only fixture has
    # no reviewed single-file bytes. Absence must still stop before loading.
    expected = FileNotFoundError if fault == 'weight_format' else (ValueError, KeyError)
    message = 'not installed' if fault == 'weight_format' else None
    with pytest.raises(expected, match=message):
        runtime.run(source)
    assert not runtime.events


@pytest.mark.parametrize('fault', ['hook', 'packed'])
def test_actual_installed_hook_and_bfl_storage_refuse_before_peft(runtime, tmp_path, monkeypatch, fault):
    from diffusers import Flux3Transformer2DModel
    runtime.faults[fault] = True
    injections = []
    original = Flux3Transformer2DModel.load_lora_adapter
    def observe(self, *args, **kwargs):
        injections.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Flux3Transformer2DModel, 'load_lora_adapter', observe)
    with pytest.raises(RuntimeError, match='Hooked|Packed BFL') as failure:
        runtime.run(source_graph(PAIRS[0], [descriptor(tmp_path)]))
    assert isinstance(failure.value.__cause__, ValueError)
    assert not injections


def test_same_name_different_bytes_replaces_actual_owner(runtime, tmp_path):
    source = source_graph(PAIRS[0], [descriptor(tmp_path)])
    runtime.run(source)
    previous = weakref.ref(runtime.server.node_cache['models'].loader)
    original_prediction = prediction(runtime)
    replacement = descriptor(tmp_path, seed=101)
    source['nodes']['models']['params']['lora_list']['value'] = [replacement]
    runtime.run(source)
    assert previous() is None
    assert not torch.allclose(prediction(runtime), original_prediction)


def test_post_injection_failure_invalidates_owner_and_clean_retry(runtime, tmp_path, monkeypatch):
    from diffusers import Flux3Transformer2DModel
    source = source_graph(PAIRS[0], [descriptor(tmp_path)])
    original = Flux3Transformer2DModel.load_lora_adapter
    with monkeypatch.context() as fault:
        def fail(self, *args, **kwargs):
            original(self, *args, **kwargs)
            raise RuntimeError('UNIT failure after real PEFT injection')
        fault.setattr(Flux3Transformer2DModel, 'load_lora_adapter', fail)
        with pytest.raises(RuntimeError, match='UNIT failure'):
            runtime.run(source)
    node = runtime.server.node_cache['models']
    assert node._native_disk_lora_state['componentComplete'] is False and node._cache_valid is False
    previous = weakref.ref(node.loader)
    node = None
    runtime.run(source)
    assert previous() is None and runtime.server.node_cache['models']._native_disk_lora_state['componentComplete'] is True
