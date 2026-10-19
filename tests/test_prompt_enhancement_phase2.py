"""PUP phase 2: task-owned preprocessing, bounded capture and old schema parity.

The neural fixture uses random local weights, never trained-model quality proof.
"""
from copy import deepcopy
import hashlib
import json
from unittest.mock import Mock

import pytest

import test_prompt_enhancement as stage_one
from modiff import prompt_enhancement as adapter
from modiff.native_component_lora import native_component_graph_scope
from modules.HuggingFaceTransformers import main as nodes

actual_pup = stage_one.actual_pup


def rgb_fingerprint(image):
    header = json.dumps({'mode': image.mode, 'width': image.width, 'height': image.height},
                        sort_keys=True, separators=(',', ':')).encode() + b'\n'
    return 'sha256:' + hashlib.sha256(header + image.tobytes()).hexdigest()


def test_owned_stop_precedes_image_normalization_and_template_render(actual_pup, monkeypatch):
    handle, _ = actual_pup
    node = nodes.GenerateImageVideoText('pup-precancel')
    graph, task, _ = stage_one.owned_generation(monkeypatch, node)
    task['interrupt_requested'] = True
    normalize = Mock(side_effect=AssertionError('Stopped task reached image normalization'))
    render = Mock(side_effect=AssertionError('Stopped task reached chat rendering'))
    generate = Mock(side_effect=AssertionError('Stopped task reached model generation'))
    monkeypatch.setattr(nodes, '_normalized_optional_images', normalize)
    monkeypatch.setattr(handle['preprocessor'], 'apply_chat_template', render)
    monkeypatch.setattr(handle['model'], 'generate', generate)
    with native_component_graph_scope(graph, stage_one.unit_source_fingerprint()), pytest.raises(RuntimeError) as error:
        node(model=handle, prompt='A red cube.', operation='enhance_prompt', enhancement_max_new_tokens=2)
    assert isinstance(error.value.__cause__, InterruptedError)
    assert not normalize.called and not render.called and not generate.called


@pytest.mark.parametrize('field', ['operation', 'enhancement_max_new_tokens', 'layout_conditioning'])
def test_actual_standard_registry_and_recipe_keep_new_controls_hidden(field):
    import modules
    registered = modules.MODULE_MAP['modules.HuggingFaceTransformers']['GenerateImageVideoText']['params'][field]
    overrides, values = nodes.image_text_operation_schema('AutoModelForImageTextToText', 'image_to_text', 'GenerateImageVideoText')
    effective = {**registered, **overrides.get(field, {})}
    assert effective.get('hidden') is True, f'{field} is globally visible in an ordinary caption recipe'
    assert values.get('operation', 'standard') == 'standard'


@pytest.mark.parametrize('version', [True, 1.0])
def test_descriptor_schema_version_requires_exact_integer(actual_pup, monkeypatch, tmp_path, version):
    handle, _ = actual_pup
    contract = json.loads(adapter.DESCRIPTOR.read_text())
    contract['schemaVersion'] = version
    replacement = tmp_path / 'malformed-version.json'
    replacement.write_text(json.dumps(contract), encoding='utf-8')
    monkeypatch.setattr(adapter, 'DESCRIPTOR', replacement)
    with pytest.raises(ValueError, match='declaration'):
        adapter.selected_contract(handle['model'], handle['preprocessor'], handle['receipt'])


@pytest.mark.parametrize('name', [None, [], 'AbsentReviewedExport'])
def test_malformed_or_missing_declared_export_has_actionable_refusal(monkeypatch, tmp_path, name):
    contract = json.loads(adapter.DESCRIPTOR.read_text())
    contract['modelClass'] = name
    replacement = tmp_path / 'malformed-export.json'
    replacement.write_text(json.dumps(contract), encoding='utf-8')
    monkeypatch.setattr(adapter, 'DESCRIPTOR', replacement)
    receipt = {'source': {'kind': 'hub', 'repository': contract['repository'], 'revision': contract['revision']}}
    with pytest.raises(ValueError, match='declaration|reviewed.*API'):
        adapter.selected_contract(object(), object(), receipt)


def test_actual_rendered_input_and_ordered_media_need_bounded_consumed_projection(actual_pup, monkeypatch):
    from PIL import Image
    handle, _ = actual_pup
    processor = handle['preprocessor']
    actual_render = processor.apply_chat_template
    rendered = []
    def observe(messages, **kwargs):
        value = actual_render(messages, **kwargs)
        rendered.append(value)
        return value
    monkeypatch.setattr(processor, 'apply_chat_template', observe)
    images = [Image.new('RGB', (256, 256), color) for color in ('red', 'blue')]
    original = deepcopy(images)
    node = nodes.GenerateImageVideoText('pup-consumed')
    node(model=handle, prompt='Use the second image behind the first.', images=images,
         operation='enhance_prompt', enhancement_max_new_tokens=2, min_new_tokens=2)
    record = node._execution_input_record['fields']
    expected_rendered = 'sha256:' + hashlib.sha256(rendered[-1].encode()).hexdigest()
    expected_sources = [rgb_fingerprint(image) for image in images]
    # These finite aliases are proposed. Source-only RGB fingerprints must not
    # be represented as the VAE-issued reference-conditioning authority IDs.
    assert record.get('rendered_prompt_sha256', {}).get('value') == expected_rendered
    assert record.get('reference_source_sha256', {}).get('value') == expected_sources
    assert record.get('enhancement_max_new_tokens', {}).get('value') == 2
    assert all(image.tobytes() == previous.tobytes() for image, previous in zip(images, original))


def test_actual_rendered_tensor_capture_is_computed_bounded_and_scope_local(actual_pup, monkeypatch):
    import sys
    from PIL import Image
    from modiff.execution_input_provenance import capture_generation_inputs, _field_names
    handle, _ = actual_pup
    processor = handle['preprocessor']
    real_call = type(processor).__call__
    processed = []
    def observe(self, **kwargs):
        batch = real_call(self, **kwargs)
        processed.append({key: value.clone() for key, value in batch.items()})
        return batch
    monkeypatch.setattr(type(processor), '__call__', observe)
    images = [Image.new('RGB', (256, 256), 'red'), Image.new('RGB', (256, 512), 'blue')]
    node = nodes.GenerateImageVideoText('pup-actual-consumed')
    node(model=handle, prompt='Keep both images.', images=images, operation='enhance_prompt',
         enhancement_max_new_tokens='2', min_new_tokens=2, rendered_prompt_sha256='caller-forgery',
         reference_source_sha256=['caller-forgery'])
    fields = node._execution_input_record['fields']
    projection = node.output['result']['consumedInput']
    assert fields['enhancement_max_new_tokens']['value'] == 2
    assert fields['reference_source_sha256']['value'] == [rgb_fingerprint(image) for image in images]
    assert [ref['index'] for ref in projection['references']] == [0, 1]
    assert projection['processedImageGrid'] == processed[-1]['image_grid_thw'].tolist()
    assert projection['renderedPrompt'].endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    for label, key in [('rendered_input_ids', 'input_ids'), ('processed_pixel', 'pixel_values'), ('processed_grid', 'image_grid_thw')]:
        value = processed[-1][key]
        header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape), 'byteOrder': sys.byteorder},
                            sort_keys=True, separators=(',', ':')).encode() + b'\n'
        import torch
        expected = 'sha256:' + hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        assert fields[label + '_sha256']['value'] == expected
    assert len(json.dumps(projection).encode()) < 256 * 1024
    assert fields['prompt']['value'] == 'Keep both images.'
    aliases = _field_names(node._execution_input_record)
    assert aliases['reference_source_sha256'] == 'referenceSourceSha256'
    for module, action in [('HuggingFaceTransformers', 'PredictDepth'), ('ModularDiffusers', 'EncodePrompt')]:
        record = capture_generation_inputs('other', {'module': module, 'action': action}, {'rendered_prompt_sha256': 'x'})
        assert record['fields'] == {}


def test_rgb_fingerprint_binds_geometry_and_order():
    from PIL import Image
    first, second = Image.new('RGB', (1, 2), 'red'), Image.new('RGB', (2, 1), 'red')
    assert first.tobytes() == second.tobytes()
    assert rgb_fingerprint(first) != rgb_fingerprint(second)


@pytest.mark.parametrize('key,value', [('maxReferences', True), ('modelWeight', None),
    ('modelWeight', {'path': '../model.safetensors', 'byteSize': 3, 'sha256': '0' * 64}),
    ('revision', 'main'), ('repository', []), ('processorClass', 42), ('public', True)])
def test_descriptor_rejects_malformed_scope_before_api_access(monkeypatch, tmp_path, key, value):
    contract = json.loads(adapter.DESCRIPTOR.read_text())
    contract[key] = value
    replacement = tmp_path / 'descriptor.json'
    replacement.write_text(json.dumps(contract), encoding='utf-8')
    monkeypatch.setattr(adapter, 'DESCRIPTOR', replacement)
    with pytest.raises(ValueError, match='declaration'):
        adapter.selected_contract(object(), object(), {'source': {}})


def test_declared_export_dependency_error_is_not_disguised(monkeypatch):
    import transformers
    original = transformers.__getattr__
    monkeypatch.delattr(transformers, 'Qwen3_5ForConditionalGeneration', raising=False)
    def broken(name):
        if name == 'Qwen3_5ForConditionalGeneration':
            raise ImportError('actual declared export dependency error')
        return original(name)
    monkeypatch.setattr(transformers, '__getattr__', broken)
    contract = json.loads(adapter.DESCRIPTOR.read_text())
    receipt = {'source': {'kind': 'hub', 'repository': contract['repository'], 'revision': contract['revision']}}
    with pytest.raises(ImportError, match='actual declared export'):
        adapter.selected_contract(object(), object(), receipt)


def test_standard_operation_projection_is_exact_legacy_projection():
    import modules
    current = deepcopy(modules.MODULE_MAP)
    legacy = deepcopy(current)
    for field in nodes.PROMPT_ENHANCEMENT_RECIPE_FIELDS:
        legacy['modules.HuggingFaceTransformers']['GenerateImageVideoText']['params'].pop(field)
    assert nodes.get_image_text_operation_contracts(current) == nodes.get_image_text_operation_contracts(legacy)
    overlay, values = nodes.prompt_enhancement_operation_schema()
    assert set(overlay) == set(nodes.PROMPT_ENHANCEMENT_RECIPE_FIELDS)
    assert all(field == {'hidden': False} for field in overlay.values())
    assert values == {'operation': 'enhance_prompt'}


def test_old_role_schemas_are_exact_and_opt_in_fields_require_actual_recipe_use():
    import modules
    from modiff.studio_execution_specs import (STUDIO_EXECUTION_SPEC_DEFINITIONS, _public_spec,
                                               _execution_spec_role_params)
    key = 'modules.HuggingFaceTransformers.GenerateImageVideoText'
    definition = modules.MODULE_MAP['modules.HuggingFaceTransformers']['GenerateImageVideoText']
    original = deepcopy(definition)
    legacy = deepcopy(definition)
    for field in nodes.PROMPT_ENHANCEMENT_RECIPE_FIELDS:
        legacy['params'].pop(field)
    checked = []
    for identifier, declaration in STUDIO_EXECUTION_SPEC_DEFINITIONS.items():
        public = _public_spec(identifier, declaration)
        if not any(node_key == key for _role, node_key, _x, _y in public['roles']):
            continue
        assert _execution_spec_role_params(public, key, definition) == _execution_spec_role_params(public, key, legacy)
        checked.append(identifier)
        role = next(role for role, node_key, _x, _y in public['roles'] if node_key == key)
        for field in nodes.PROMPT_ENHANCEMENT_RECIPE_FIELDS:
            for kind in ['binding', 'connection']:
                explicit = deepcopy(public)
                if kind == 'binding':
                    explicit['bindings'] = (*explicit['bindings'], (role, field, 'hypotheticalExplicitValue'))
                else:
                    explicit['edges'] = (*explicit['edges'], ('hypotheticalSupplier', 'value', role, field))
                projected = _execution_spec_role_params(explicit, key, definition)
                assert projected[field] == definition['params'][field]
    assert checked and definition == original


def test_generic_opt_in_omission_preserves_exact_existing_weight_format_roles():
    import modules
    from modiff.studio_execution_specs import (STUDIO_EXECUTION_SPEC_DEFINITIONS, _public_spec,
                                               _execution_spec_role_params)
    key = 'modules.ModularDiffusers.ModelsLoader'
    definition = modules.MODULE_MAP['modules.ModularDiffusers']['ModelsLoader']
    original = deepcopy(definition)
    checked = 0
    for identifier, declaration in STUDIO_EXECUTION_SPEC_DEFINITIONS.items():
        public = _public_spec(identifier, declaration)
        roles = {role for role, node_key, _x, _y in public['roles'] if node_key == key}
        if not roles:
            continue
        selected = any(role in roles and field == 'weight_format' for role, field, _source in public['bindings'])
        selected |= any(role in roles and field == 'weight_format' for _source, _port, role, field in public['edges'])
        actual = _execution_spec_role_params(public, key, definition)
        assert ('weight_format' in actual) == selected
        if selected:
            assert actual['weight_format'] == definition['params']['weight_format']
        checked += 1
    assert checked > 20 and definition == original


def test_text_only_projection_has_no_borrowed_image_authority(actual_pup):
    handle, _ = actual_pup
    node = nodes.GenerateImageVideoText('pup-text-consumed')
    node(model=handle, prompt='A red cube.', operation='enhance_prompt', enhancement_max_new_tokens=2)
    fields = node._execution_input_record['fields']
    projection = node.output['result']['consumedInput']
    assert fields['reference_source_sha256']['value'] == []
    assert 'processed_pixel_sha256' not in fields and projection['processedImageGrid'] == []
    assert 'referenceId' not in json.dumps(projection)


def test_stop_during_normalization_prevents_template_processor_and_model(actual_pup, monkeypatch):
    from PIL import Image
    handle, _ = actual_pup
    node = nodes.GenerateImageVideoText('pup-normalization-stop')
    graph, task, _ = stage_one.owned_generation(monkeypatch, node)
    original = nodes._normalized_optional_images
    def stop(value):
        result = original(value)
        task['interrupt_requested'] = True
        return result
    monkeypatch.setattr(nodes, '_normalized_optional_images', stop)
    render = Mock(side_effect=AssertionError('Stop reached template rendering'))
    generate = Mock(side_effect=AssertionError('Stop reached model generation'))
    monkeypatch.setattr(handle['preprocessor'], 'apply_chat_template', render)
    monkeypatch.setattr(handle['model'], 'generate', generate)
    with native_component_graph_scope(graph, stage_one.unit_source_fingerprint()), pytest.raises(RuntimeError) as error:
        node(model=handle, prompt='A cube.', images=[Image.new('RGB', (32, 32))],
             operation='enhance_prompt', enhancement_max_new_tokens=2)
    assert isinstance(error.value.__cause__, InterruptedError)
    assert not render.called and not generate.called


def test_multi_chunk_tensor_hash_is_exact_and_honors_stop():
    from types import SimpleNamespace
    import sys
    import torch
    value = torch.arange(600_000, dtype=torch.int64)
    observer = SimpleNamespace(check=Mock())
    header = json.dumps({'dtype': str(value.dtype), 'shape': list(value.shape), 'byteOrder': sys.byteorder},
                        sort_keys=True, separators=(',', ':')).encode() + b'\n'
    expected = 'sha256:' + hashlib.sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert adapter._tensor_fingerprint(value, observer) == expected
    assert observer.check.call_count > 8
    observer.check = Mock(side_effect=[None, None, None, None, InterruptedError('owned task Stop')])
    with pytest.raises(InterruptedError, match='owned task Stop'):
        adapter._tensor_fingerprint(value, observer)
    with pytest.raises(ValueError, match='contiguous CPU'):
        adapter._tensor_fingerprint(value.reshape(600, 1000).t(), SimpleNamespace(check=Mock()))


def test_processor_grid_mismatch_is_refused_before_neural_generate(actual_pup, monkeypatch):
    from PIL import Image
    handle, _ = actual_pup
    processor = handle['preprocessor']
    original = type(processor).__call__
    def malformed(self, **kwargs):
        batch = original(self, **kwargs)
        batch['image_grid_thw'] = batch['image_grid_thw'][:0]
        return batch
    monkeypatch.setattr(type(processor), '__call__', malformed)
    generate = Mock(side_effect=AssertionError('Malformed grid reached model'))
    monkeypatch.setattr(handle['model'], 'generate', generate)
    node = nodes.GenerateImageVideoText('pup-malformed-grid')
    # The shared resource guard now validates the processed grid before the
    # later prompt-consumption projection can inspect it.
    with pytest.raises(RuntimeError, match='processed image-pixel resource bound.*actual CPU image grid') as error:
        node(model=handle, prompt='A cube.', images=[Image.new('RGB', (256, 256))],
             operation='enhance_prompt', enhancement_max_new_tokens=2)
    assert isinstance(error.value.__cause__, ValueError)
    assert not generate.called
