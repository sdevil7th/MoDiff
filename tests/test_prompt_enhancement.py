"""Optional prompt enhancement: local synthetic neural weights, actual processor.

The fixture substitutes random tiny weights at the official model factory only.
It does not establish pretrained PUP quality, Windows execution or model fit.
"""
from copy import deepcopy
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from modules.HuggingFaceTransformers.main import LoadImageTextToTextModel, GenerateImageVideoText
from utils.torch_utils import DEFAULT_DEVICE

REPOSITORY = 'black-forest-labs/flux-3-image-klein-pup'
REVISION = '3f3edb52b2c55de0d49feab28598b68f7f8e4830'


@pytest.fixture
def actual_pup(tmp_path, monkeypatch):
    capture = os.environ.get('MODIFF_PUP_PROCESSOR_FIXTURE')
    if not capture:
        pytest.skip('Explicit reviewed local PUP processor fixture is not supplied.')
    capture = Path(capture)
    authority = json.loads((capture / 'result.json').read_text(encoding='utf-8'))
    processor_path = capture / 'processor'
    assert authority['status'] == 'passed' and authority['revision'] == REVISION
    assert hashlib.sha256((processor_path / 'tokenizer.json').read_bytes()).hexdigest() == authority['tokenizerSha256']
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              Qwen3_5Config, Qwen3_5ForConditionalGeneration)
    captured = json.loads((processor_path / 'config.json').read_text(encoding='utf-8'))
    tiny = deepcopy(captured)
    tiny['dtype'] = 'float32'
    tiny['text_config'].update(dtype='float32', hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=1, num_key_value_heads=1, head_dim=32, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'])
    tiny['text_config']['rope_parameters']['mrope_section'] = [1, 1, 2]
    tiny['vision_config'].update(dtype='float32', hidden_size=32, intermediate_size=64, depth=1, num_heads=1,
                                 out_hidden_size=32, num_position_embeddings=64)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4201)
            model = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(tiny)).float().eval()
        model.set_attn_implementation('eager')
        local = tmp_path / 'explicit-synthetic-neural-weights'
        model.save_pretrained(local, safe_serialization=True)
        # Exercise the same selected closure with explicit random tiny weights.
        # Only this test's weight size/hash changes; all eight publisher metadata
        # identities stay genuine. This is not a pretrained model inventory.
        from modiff import prompt_enhancement as adapter, model_artifact_catalog as catalog
        from modiff.config import CONFIG
        hub = tmp_path / 'toy-managed-hub'
        snapshot = hub / ('models--' + REPOSITORY.replace('/', '--')) / 'snapshots' / REVISION
        snapshot.mkdir(parents=True)
        contents = deepcopy(catalog.read_model_artifact_catalog())
        inventory = next(row for row in contents['downloadInventories'] if row['repo'] == REPOSITORY)
        declaration = json.loads(adapter.DESCRIPTOR.read_bytes())
        for row in inventory['files']:
            source = local / row['path'] if row['path'] == 'model.safetensors' else processor_path / row['path']
            shutil.copyfile(source, snapshot / row['path'])
            if row['path'] == 'model.safetensors':
                row.update(byteSize=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                declaration['modelWeight'].update(byteSize=row['byteSize'], sha256=row['sha256'])
        catalog_path, descriptor_path = tmp_path / 'toy-catalog.json', tmp_path / 'toy-adapter.json'
        catalog_path.write_text(json.dumps(contents), encoding='utf-8')
        descriptor_path.write_text(json.dumps(declaration), encoding='utf-8')
        monkeypatch.setattr(catalog, 'CATALOG_PATH', catalog_path)
        monkeypatch.setattr(adapter, 'DESCRIPTOR', descriptor_path)
        monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(hub))
        monkeypatch.setitem(CONFIG.hf, 'token', None)
        original_model = AutoModelForImageTextToText.from_pretrained
        original_processor = AutoProcessor.from_pretrained
        calls = []
        def load_model(source, **kwargs):
            calls.append(('model', source, dict(kwargs)))
            assert Path(source) == snapshot and kwargs.pop('revision') == REVISION
            loaded = original_model(local, **kwargs)
            loaded.set_attn_implementation('eager')
            return loaded
        def load_processor(source, **kwargs):
            calls.append(('processor', source, dict(kwargs)))
            assert Path(source) == snapshot and kwargs.pop('revision') == REVISION
            return original_processor(source, **kwargs)
        monkeypatch.setattr(AutoModelForImageTextToText, 'from_pretrained', load_model)
        monkeypatch.setattr(AutoProcessor, 'from_pretrained', load_processor)
        loader = LoadImageTextToTextModel('pup-model')
        loader(model_id={'source': 'hub', 'value': REPOSITORY}, revision=REVISION, dtype='float32', device=DEFAULT_DEVICE)
        assert [kind for kind, *_ in calls] == ['processor', 'model']
        yield loader.output['model'], calls
    finally:
        torch.set_num_threads(threads)


def test_actual_text_only_enhancement_uses_existing_nodes_and_retains_authored_request(actual_pup):
    handle, calls = actual_pup
    authored = '  A red cube on a wooden table.  '
    node = GenerateImageVideoText('pup-generate')
    node(model=handle, prompt=authored, operation='enhance_prompt', enhancement_max_new_tokens=2, min_new_tokens=2)
    result = node.output['result']
    assert result['task'] == 'prompt-enhancement'
    assert result['authoredPrompt'] == authored
    assert result['applyRequired'] is True
    assert result['inputImageCount'] == 0 and result['generatedTokens'] == 2
    assert result['systemPromptMode'] == 't2i'
    assert node.output['text'] == result['text']
    assert all(kwargs['local_files_only'] is True and kwargs['trust_remote_code'] is False for _, _, kwargs in calls)


def test_actual_two_reference_neural_generation_preserves_order(actual_pup, monkeypatch):
    from PIL import Image
    handle, _ = actual_pup
    processor = handle['preprocessor']
    original = processor.apply_chat_template
    messages = []
    def observe(value, **kwargs):
        messages.append((deepcopy(value), dict(kwargs)))
        return original(value, **kwargs)
    monkeypatch.setattr(processor, 'apply_chat_template', observe)
    images = [Image.new('RGB', (256, 256), color) for color in ('red', 'blue')]
    node = GenerateImageVideoText('pup-two-references')
    node(model=handle, prompt='Combine the first and second images.', operation='enhance_prompt', images=images,
         enhancement_max_new_tokens=2, min_new_tokens=2)
    result = node.output['result']
    assert result['inputImageCount'] == 2 and result['systemPromptMode'] == 'i2i'
    assert result['generatedTokens'] == 2 and result['semanticFidelityVerified'] is False
    assert [item['type'] for item in messages[-1][0][1]['content']] == ['image', 'image', 'text']
    assert messages[-1][1]['enable_thinking'] is False
    assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255)]


def test_actual_processor_renders_all_four_pinned_modes_and_refuses_drift(actual_pup, monkeypatch, tmp_path):
    from PIL import Image
    from modiff import prompt_enhancement as adapter
    handle, _ = actual_pup
    model, processor, receipt = handle['model'], handle['preprocessor'], handle['receipt']
    contract, prompts = adapter.selected_contract(model, processor, receipt)
    assert contract['maxNewTokens'] == 3584 and set(prompts) == {'t2i', 'i2i', 't2i_bbox', 'i2i_bbox'}
    layout = {'schemaVersion': 1, 'coordinateSpace': 'normalized_xyxy', 'regions': [
        {'id': 'cube_1', 'description': 'A red cube.', 'targetBox': [.1, .2, .4, .5]}]}
    baseline = deepcopy(layout)
    for count, supplied, expected in [(0, None, 't2i'), (2, None, 'i2i'), (10, None, 'i2i'),
                                      (0, layout, 't2i_bbox'), (2, layout, 'i2i_bbox')]:
        images = [Image.new('RGB', (256, 256), 'red')] * count
        mode, rendered, parsed = adapter.prepare_request(processor, prompts, 'Cube <cube_1>.', images, supplied)
        assert mode == expected and prompts[expected] in rendered
        assert (parsed is None) == (supplied is None)
        kwargs = {'text': rendered, 'return_tensors': 'pt'}
        if images:
            kwargs['images'] = images
        batch = processor(**kwargs)
        assert batch['input_ids'].shape[0] == 1
        assert ('pixel_values' in batch) == bool(count)
        if supplied:
            assert ('"tgt_bbox"' if count else '"bbox"') + ':[200,100,500,400]' in rendered
    assert layout == baseline
    with pytest.raises(ValueError, match='at most ten'):
        adapter.prepare_request(processor, prompts, 'A cube.', [Image.new('RGB', (16, 16))] * 11, None)
    with_source = deepcopy(layout)
    with_source['regions'][0]['source'] = {'referenceIndex': 0, 'box': [.1, .2, .4, .5]}
    mode, rendered, parsed = adapter.prepare_request(processor, prompts, 'Cube <cube_1>.', [Image.new('RGB', (16, 16))], with_source)
    assert mode == 'i2i_bbox' and parsed == with_source and '"src_bbox":[200,100,500,400]' in rendered
    for source in ({'kind': 'hub', 'repository': REPOSITORY, 'revision': '0' * 40},
                   {'kind': 'local', 'path': 'not-authority'}):
        changed = deepcopy(receipt)
        changed['source'] = source
        with pytest.raises(ValueError, match='immutable'):
            adapter.selected_contract(model, processor, changed)
    with pytest.raises(ValueError, match='native Transformers'):
        adapter.selected_contract(object(), processor, receipt)
    altered = tmp_path / 'changed-prompts.json'
    altered.write_bytes(adapter.SYSTEM_PROMPTS.read_bytes() + b' ')
    monkeypatch.setattr(adapter, 'SYSTEM_PROMPTS', altered)
    with pytest.raises(ValueError, match='system prompts have changed'):
        adapter.selected_contract(model, processor, receipt)


def test_scoped_token_limit_and_standard_path_preserve_existing_controls():
    from modules.HuggingFaceTransformers.main import _generation_controls
    with pytest.raises(ValueError, match='max_new_tokens'):
        _generation_controls({'max_new_tokens': 3584})
    assert _generation_controls({'max_new_tokens': 3584}, maximum_new_tokens=3584)['max_new_tokens'] == 3584
    assert GenerateImageVideoText.params['max_new_tokens']['max'] == 2048
    assert GenerateImageVideoText.params['enhancement_max_new_tokens']['max'] == 3584
    assert GenerateImageVideoText.params['operation']['default'] == 'standard'
    descriptor = json.loads((Path(__file__).resolve().parents[1] / 'data/prompt-enhancement-adapters.v1.json').read_text())
    assert all(descriptor[name] is False for name in ('public', 'autoEligible', 'templateEligible', 'liveProof'))


def unit_source_fingerprint():
    from modiff.backend_source_identity import process_backend_source_identity
    return process_backend_source_identity()['fingerprint']


def owned_generation(monkeypatch, node, *, sid=None):
    import importlib
    graph = {'sid': sid, 'nodes': {node.node_id: {
        'module': 'modules.HuggingFaceTransformers', 'action': 'GenerateImageVideoText', 'params': {}}}}
    graph['paths'] = [[node.node_id]]
    task = {'task_id': 'pup-task', 'attempt_index': 0, 'args': [graph], 'sid': sid,
            'current_node': node.node_id, 'interrupt_requested': False}
    server = SimpleNamespace(current_task=task, node_cache={node.node_id: node},
                             record_node_progress=Mock(side_effect=lambda value: value), queue_message=Mock())
    import modules
    server.modules = modules.MODULE_MAP
    server.instance = 'explicit-pup-observer-unit-worker'
    server.backend_source_identity = {'fingerprint': unit_source_fingerprint()}
    node._sid = sid
    monkeypatch.setattr(importlib.import_module('modiff.NodeBase'), '_server', lambda: server)
    return graph, task, server


def test_actual_stop_before_and_during_neural_generation_is_task_owned(actual_pup, monkeypatch):
    from modiff.native_component_lora import native_component_graph_scope
    handle, _ = actual_pup
    node = GenerateImageVideoText('pup-stop')
    graph, task, server = owned_generation(monkeypatch, node)
    real_generate = handle['model'].generate
    calls = []
    def observe(**kwargs):
        calls.append(kwargs)
        return real_generate(**kwargs)
    monkeypatch.setattr(handle['model'], 'generate', observe)
    task['interrupt_requested'] = True
    with native_component_graph_scope(graph, unit_source_fingerprint()), pytest.raises(RuntimeError) as error:
        node(model=handle, prompt='A red cube.', operation='enhance_prompt', enhancement_max_new_tokens=2)
    assert isinstance(error.value.__cause__, InterruptedError) and not calls
    task['interrupt_requested'] = False
    def stop_after_forward(*_args):
        task['interrupt_requested'] = True
    hook = handle['model'].register_forward_hook(stop_after_forward)
    try:
        with native_component_graph_scope(graph, unit_source_fingerprint()), pytest.raises(RuntimeError) as error:
            node(model=handle, prompt='A red cube.', operation='enhance_prompt', enhancement_max_new_tokens=2)
        assert isinstance(error.value.__cause__, InterruptedError) and len(calls) == 1
    finally:
        hook.remove()
    assert not server.queue_message.called  # API task with no websocket target.


@pytest.mark.parametrize('fault', ['replacement', 'different-graph', 'different-node', 'none',
                                  'empty-args', 'malformed-nodes', 'changed-action'])
def test_observer_never_borrows_unrelated_task_or_stale_node_interrupt(monkeypatch, fault):
    from modiff.native_component_lora import native_component_graph_scope
    from modiff.prompt_enhancement import _OwnedGeneration
    node = SimpleNamespace(node_id='pup', _sid=None, _interrupt=True, progress=Mock())
    graph, task, server = owned_generation(monkeypatch, node)
    with native_component_graph_scope(graph, unit_source_fingerprint()):
        observer = _OwnedGeneration(node)
        observer.check()  # stale node flag does not cancel this task.
        if fault == 'replacement':
            server.current_task = dict(task)
        elif fault == 'different-graph':
            task['args'] = [deepcopy(graph)]
        elif fault == 'different-node':
            server.node_cache[node.node_id] = object()
        elif fault == 'empty-args':
            task['args'] = []
        elif fault == 'malformed-nodes':
            graph['nodes'] = None
        elif fault == 'changed-action':
            graph['nodes'][node.node_id]['action'] = 'Other'
        else:
            server.current_task = None
        with pytest.raises(InterruptedError, match='no longer belongs'):
            observer.check()
        unbound = _OwnedGeneration(node)
        unbound.progress(1, 2)
    assert not node.progress.called and not server.queue_message.called


def test_owned_progress_is_indeterminate_and_throttled(monkeypatch):
    from modiff.native_component_lora import native_component_graph_scope
    from modiff.prompt_enhancement import _OwnedGeneration
    node = SimpleNamespace(node_id='pup', _sid='pup-session', _interrupt=False, progress=Mock())
    graph, _, _ = owned_generation(monkeypatch, node, sid='pup-session')
    with native_component_graph_scope(graph, unit_source_fingerprint()):
        observer = _OwnedGeneration(node)
        clock = iter([observer.started, observer.started + .1, observer.started + .3])
        observer.clock = lambda: next(clock)
        observer.progress(0, 10, immediate=True)
        observer.progress(1, 10)
        observer.progress(2, 10)
    assert node.progress.call_count == 2
    assert all(call.args == (-1,) and call.kwargs['current_step'] in (0, 2)
               for call in node.progress.call_args_list)


def test_invalid_selected_controls_prefix_and_layout_output_fail_closed(actual_pup, monkeypatch):
    handle, _ = actual_pup
    node = GenerateImageVideoText('pup-invalid')
    common = dict(model=handle, prompt='A cube <cube_1>.', operation='enhance_prompt', enhancement_max_new_tokens=2)
    for change, message in [(dict(video=[]), 'images and text only'),
                            (dict(use_chat_template=False), 'chat template'),
                            (dict(do_sample=True), 'greedy'),
                            (dict(enhancement_max_new_tokens=3585), 'enhancement_max_new_tokens')]:
        with pytest.raises(RuntimeError, match=message):
            node(**{**common, **change})
    monkeypatch.setattr(handle['model'], 'generate', lambda **_kwargs: torch.tensor([[1, 2]]))
    with pytest.raises(RuntimeError, match='input token prefix'):
        node(**common)
    layout = {'schemaVersion': 1, 'coordinateSpace': 'normalized_xyxy', 'regions': [
        {'id': 'cube_1', 'description': 'A red cube.', 'targetBox': [.1, .2, .4, .5]}]}
    processor = handle['preprocessor']
    def diagnostic_layout(**kwargs):
        proposed = 'Cube <cube_1> [{"id":"cube_1","bbox":[0,0,1,1],"desc":"changed"}]'
        tokens = processor.tokenizer(proposed, return_tensors='pt', add_special_tokens=False)['input_ids']
        return torch.cat([kwargs['input_ids'], tokens], dim=1)
    monkeypatch.setattr(handle['model'], 'generate', diagnostic_layout)
    node(**{**common, 'enhancement_max_new_tokens': 256, 'layout_conditioning': layout})
    assert node.output['text'] is None
    assert node.output['result']['layoutOutputValidated'] is False
    assert node.output['result']['layoutConditioning'] == layout
    assert 'applicationBlockedReason' in node.output['result']
