"""Finite image/text quantization contracts; no GPU or pretrained-fit claims."""
from types import SimpleNamespace

import pytest
import torch

import test_prompt_enhancement as pup

actual_pup = pup.actual_pup


def test_existing_enhancement_runtime_minimum_uses_selected_token_limit():
    """Execution contract only: ordinary UI min-token metadata remains 2048."""
    from modules.HuggingFaceTransformers import main as nodes
    maximum = 3584
    assert nodes._generation_controls({'max_new_tokens': maximum, 'min_new_tokens': maximum},
        maximum_new_tokens=maximum)['min_new_tokens'] == maximum
    with pytest.raises(ValueError, match='min_new_tokens'):
        nodes._generation_controls({'max_new_tokens': maximum, 'min_new_tokens': maximum + 1},
            maximum_new_tokens=maximum)
    with pytest.raises(ValueError, match='min_new_tokens'):
        nodes._generation_controls({'max_new_tokens': 2048, 'min_new_tokens': maximum})
    assert nodes.GenerateImageVideoText.params['min_new_tokens']['max'] == 2048


def test_pinned_image_text_nf4_reaches_existing_official_factory_contract(actual_pup, monkeypatch):
    """Real tiny model/processor, substituted quantized factory, no BnB kernel."""
    import transformers
    from modules.HuggingFaceTransformers import main as nodes
    from modiff import transformers_image_text_resources as resources
    handle, _calls = actual_pup
    captured = {}
    monkeypatch.setattr(nodes, '_normalized_device', lambda _value: 'cuda:0')
    def config(**kwargs):
        captured['config'] = kwargs
        return SimpleNamespace(**kwargs)
    def factory(source, **kwargs):
        captured['factory'] = {'source': str(source), **kwargs}
        return handle['model']
    monkeypatch.setattr(transformers, 'BitsAndBytesConfig', config)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained', factory)
    # This test proves factory arguments only; a separate real-model negative
    # control below proves a forged NF4 receipt cannot replace native state.
    monkeypatch.setattr(resources, 'validate_image_text_nf4_model',
        lambda model, recipe: captured.update(validatedModel=model is handle['model'], validatedRecipe=recipe))
    result = nodes.LoadImageTextToTextModel('unit-nf4-contract').execute(
        model_id={'source': 'hub', 'value': pup.REPOSITORY}, revision=pup.REVISION,
        dtype='bfloat16', device='cuda:0', quantization_mode='bnb_4bit')
    assert captured['config'] == {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': torch.bfloat16,
        'llm_int8_skip_modules': ['lm_head', 'model.visual']}
    assert captured['validatedModel'] is True
    assert captured['factory']['device_map'] == {'': 'cuda:0'}
    assert captured['factory']['trust_remote_code'] is False
    assert captured['factory']['local_files_only'] is True
    assert captured['factory']['use_safetensors'] is True
    assert captured['factory']['weights_only'] is True
    assert result['receipt']['runtime']['quantization'] == {
        'mode': 'bnb_4bit', 'quantType': 'nf4', 'doubleQuant': True,
        'computeDtype': 'bfloat16', 'skipModules': ['lm_head', 'model.visual']}


def test_image_text_nf4_requires_current_optional_runtime_activation_without_factories():
    from modiff.optional_runtime_execution import loader_optional_runtime_requirement
    from modiff.optional_runtimes import TRANSFORMERS_MAIN_PEFT_BITSANDBYTES_RUNTIME_PROFILE_ID
    value = loader_optional_runtime_requirement('modules.HuggingFaceTransformers',
        'LoadImageTextToTextModel', {'model_id': {'source': 'hub', 'value': pup.REPOSITORY},
            'revision': pup.REVISION, 'dtype': 'bfloat16', 'device': 'cuda:0',
            'quantization_mode': 'bnb_4bit'}, catalog_resolver=lambda: {})
    assert value['requiredNow'] is True
    assert value['profileIds'] == [TRANSFORMERS_MAIN_PEFT_BITSANDBYTES_RUNTIME_PROFILE_ID]
    assert value['state'] != 'active'


def test_actual_dense_model_cannot_claim_nf4_from_a_receipt(actual_pup):
    from modiff.transformers_image_text_resources import image_text_nf4_policy, validate_image_text_nf4_model
    recipe = image_text_nf4_policy({'source': 'hub', 'value': pup.REPOSITORY},
        pup.REVISION, 'bfloat16', 'cuda:0')
    with pytest.raises(ValueError, match='actual native quantized model state'):
        validate_image_text_nf4_model(actual_pup[0]['model'], recipe)


def test_nf4_target_census_matches_actual_official_tiny_module_types(actual_pup):
    from modiff.transformers_image_text_resources import nf4_linear_weight, nf4_parameter_storage, _expected_nf4_linear_shapes
    model = actual_pup[0]['model']
    named = dict(model.named_modules())
    tensors = {}
    selected = []
    for name, value in model.state_dict().items():
        tensors[name] = {'shape': list(value.shape), 'dtype': 'F32' if value.is_floating_point() else 'I64'}
        if nf4_linear_weight(name, list(value.shape)):
            assert type(named[name.removesuffix('.weight')]) is torch.nn.Linear
            selected.append(name)
    expected = [name + '.weight' for name, module in named.items() if type(module) is torch.nn.Linear
                and name.startswith('model.language_model.layers.')]
    assert set(selected) == set(expected) and selected
    assert _expected_nf4_linear_shapes(model.config.to_dict()) == {
        name: tensors[name]['shape'] for name in selected}
    assert not nf4_linear_weight('model.language_model.embed_tokens.weight', [100, 32])
    assert not nf4_linear_weight('lm_head.weight', [100, 32])
    assert not nf4_linear_weight('model.visual.blocks.0.mlp.linear_fc1.weight', [32, 64])
    storage = nf4_parameter_storage(tensors)
    assert storage['quantizedLinearCount'] == len(expected)
    assert storage['denseParameterBytes'] > 0 and storage['quantizedParameterBytes'] > 0
    assert storage['parameterStorageBytes'] == sum(storage[name] for name in (
        'denseParameterBytes', 'quantizedParameterBytes', 'quantizationStateReserveBytes'))


def test_inspector_refuses_publisher_config_with_substituted_incompatible_tiny_weights(actual_pup, monkeypatch):
    """Same immutable-shaped test selector is insufficient without shape join."""
    import transformers
    from modiff.transformers_image_text_resources import inspect_image_text_nf4_storage
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained',
        lambda *_args, **_kwargs: pytest.fail('resource inspection constructed a model'))
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained',
        lambda *_args, **_kwargs: pytest.fail('resource inspection constructed a processor'))
    # actual_pup intentionally preserves publisher metadata while substituting
    # four-layer random weights. Resource inspection must detect that mismatch.
    with pytest.raises(ValueError, match='targets/shapes differ'):
        inspect_image_text_nf4_storage({'source': 'hub', 'value': pup.REPOSITORY}, pup.REVISION)


def test_read_only_storage_inspector_accepts_faithful_synthetic_config_producer(actual_pup, monkeypatch):
    """Explicit local fixture provenance; no publisher, GPU or fit claim."""
    import hashlib
    import json
    from pathlib import Path
    import transformers
    from modiff import model_artifact_catalog as catalog
    from modiff.config import CONFIG
    from modiff.transformers_image_text_resources import inspect_image_text_nf4_storage, image_text_working_storage
    snapshot = Path(CONFIG.hf['cache_dir']) / ('models--' + pup.REPOSITORY.replace('/', '--')) / 'snapshots' / pup.REVISION
    config = snapshot / 'config.json'
    # Use the actual official model's serializer, not fabricated catalog hashes.
    actual_pup[0]['model'].config.to_json_file(config, use_diff=False)
    contents = catalog.read_model_artifact_catalog()
    inventory = next(row for row in contents['downloadInventories'] if row['repo'] == pup.REPOSITORY)
    entry = next(row for row in inventory['files'] if row['path'] == 'config.json')
    entry.update(byteSize=config.stat().st_size, sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    catalog.CATALOG_PATH.write_text(json.dumps(contents), encoding='utf-8')
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained',
        lambda *_args, **_kwargs: pytest.fail('resource inspection constructed a model'))
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained',
        lambda *_args, **_kwargs: pytest.fail('resource inspection constructed a processor'))
    observed = inspect_image_text_nf4_storage({'source': 'hub', 'value': pup.REPOSITORY}, pup.REVISION)
    assert observed['quantizedLinearCount'] > 0 and observed['inferencePeakMeasured'] is False
    assert observed['configSha256'] == entry['sha256']
    demand = image_text_working_storage(observed['config'], observed,
        input_tokens=4096, image_pixels=1048576, references=10, new_tokens=3584)
    assert demand['inferencePeakMeasured'] is False and demand['requiresRuntimeHeadroomPolicy'] is True
    assert demand['requirements']['vramBytes'] > observed['parameterStorageBytes']
    assert demand['requirements']['systemRamBytes'] > observed['largestSourceTensorBytes']
    with pytest.raises(ValueError, match='no reviewed working envelope'):
        image_text_working_storage(observed['config'], observed,
            input_tokens=32768, image_pixels=1048576, references=10, new_tokens=3584)
    # Exercise the existing complete planner with an explicitly controlled
    # policy producer. The production policy below remains closed; this tiny
    # local serializer fixture establishes arithmetic/ownership plumbing only.
    import modules
    import test_ancillary_execution as ancillary
    from modiff import transformers_image_text_resources as resources
    from modiff.workflow_auto_resource import build_workflow_auto_plan
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['runtimeHints']['resourceMode'] = 'auto'
    graph['nodes']['provider']['params'].update(dtype={'value': 'bfloat16'},
        device={'value': 'cuda:0'}, quantization_mode={'value': 'bnb_4bit'})
    graph['nodes']['proposal']['params'].update(max_input_tokens={'value': 4096},
        max_image_pixels={'value': 1048576}, enhancement_max_new_tokens={'value': 3584})
    def controlled_envelope(observation, **bounds):
        if observation is None:
            return None  # Test-only structural gate; not a usable budget.
        assert observation['configSha256'] == entry['sha256']
        return image_text_working_storage(observation['config'], observation, **bounds)
    monkeypatch.setattr(resources, 'reviewed_image_text_auto_envelope', controlled_envelope)
    hardware = {'systemMemory': {'totalBytes': 64 * 1024**3, 'availableBytes': 60 * 1024**3},
        'accelerator': {'kind': 'cuda', 'totalBytes': 24 * 1024**3, 'freeBytes': 22 * 1024**3},
        'offloadDisk': {'freeBytes': 100 * 1024**3}}
    plan = build_workflow_auto_plan(graph, runtime_fingerprint={}, local_models=[],
        data_dir=str(CONFIG.paths['data']), hardware=hardware)
    assert plan['canAutoRun'] is True and not plan['issues'] and plan['patches'] == []
    assert len(plan['loaders']) == 1 and plan['loaders'][0]['nodeId'] == 'provider'
    assert plan['loaders'][0]['consumers'] == ['proposal']
    assert plan['requirements']['vramBytes'] >= plan['loaders'][0]['requirements']['vramBytes'] + 2 * 1024**3


@pytest.mark.parametrize('changed', [
    {'dtype': 'float32'}, {'device': 'cpu:0'}, {'device': 'cuda:1'},
    {'revision': 'main'}, {'selection': {'source': 'hub', 'value': 'foreign/model'}},
])
def test_image_text_nf4_recipe_does_not_admit_nearby_unsupported_cases(changed):
    from modiff.transformers_image_text_resources import image_text_nf4_policy
    values = {'selection': {'source': 'hub', 'value': pup.REPOSITORY},
              'revision': pup.REVISION, 'dtype': 'bfloat16', 'device': 'cuda:0', **changed}
    with pytest.raises(ValueError, match='exact reviewed immutable artifact'):
        image_text_nf4_policy(**values)


def test_actual_processor_token_bound_refuses_before_neural_generation(actual_pup):
    from modules.HuggingFaceTransformers.main import GenerateImageVideoText
    node = GenerateImageVideoText('unit-input-bound')
    with pytest.raises(ValueError, match='input.token.*bound'):
        node.execute(model=actual_pup[0], prompt='A tiny red cube.', operation='enhance_prompt',
            enhancement_max_new_tokens=2, min_new_tokens=2, max_input_tokens=1)


def test_reference_pixel_bound_refuses_before_processor_work(actual_pup, monkeypatch):
    from PIL import Image
    from modules.HuggingFaceTransformers.main import GenerateImageVideoText
    processor = actual_pup[0]['preprocessor']
    monkeypatch.setattr(type(processor), '__call__',
        lambda *_args, **_kwargs: pytest.fail('raw reference budget must refuse before processor work'))
    with pytest.raises(ValueError, match='image.pixel.*bound'):
        GenerateImageVideoText('unit-pixel-bound').execute(model=actual_pup[0],
            prompt='A tiny red cube.', images=[Image.new('RGB', (512, 512))], operation='enhance_prompt',
            enhancement_max_new_tokens=2, min_new_tokens=2, max_image_pixels=65536)


def test_read_only_image_text_i2i_auto_recipe_stays_closed_before_artifact_work(monkeypatch, tmp_path):
    """Recognize the exact owner without publishing unmeasured fit authority."""
    import modules
    import transformers
    import test_ancillary_execution as ancillary
    from modiff.workflow_auto_resource import build_workflow_auto_plan
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP, 1)
    graph['runtimeHints']['resourceMode'] = 'auto'
    graph['nodes']['provider']['params'].update(dtype={'value': 'bfloat16'},
        device={'value': 'cuda:0'}, quantization_mode={'value': 'bnb_4bit'})
    graph['nodes']['proposal']['params'].update(max_input_tokens={'value': 4096},
        max_image_pixels={'value': 1048576}, enhancement_max_new_tokens={'value': 3584})
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained',
        lambda *_args, **_kwargs: pytest.fail('Auto inspection constructed a model'))
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained',
        lambda *_args, **_kwargs: pytest.fail('Auto inspection constructed a processor'))
    hardware = {'systemMemory': {'totalBytes': 64 * 1024**3, 'availableBytes': 60 * 1024**3},
        'accelerator': {'kind': 'cuda', 'totalBytes': 24 * 1024**3, 'freeBytes': 22 * 1024**3},
        'offloadDisk': {'freeBytes': 100 * 1024**3}}
    plan = build_workflow_auto_plan(graph, runtime_fingerprint={}, local_models=[],
        data_dir=str(tmp_path), hardware=hardware)
    assert plan['canAutoRun'] is False and plan['loaders'] == []
    assert any('text-only request' in issue for issue in plan['issues'])


@pytest.mark.parametrize('available,refused', [(6, True), (7, False)])
def test_actual_capacity_recheck_keeps_host_loading_storage_additive_to_headroom(available, refused):
    """Synthetic capacity arithmetic only; no artifact or hardware fit proof."""
    from modiff.workflow_auto_lifecycle import assert_next_owner_capacity
    gib = 1024**3
    owner = {'ownerId': 'incoming-image-text', 'device': 'cuda:0',
        'requirements': {'systemRamBytes': 3 * gib, 'vramBytes': 0, 'diskFreeBytes': 0},
        'capacityRequirements': {}, 'workingMemoryPolicy': 'runtime_headroom_policy',
        'parameterResidencyRequirements': {'systemRamBytes': 0, 'vramBytes': 0},
        'parameterLoadingRequirements': {'systemRamBytes': 3 * gib, 'vramBytes': 0}}
    hardware = {'systemMemory': {'totalBytes': 32 * gib, 'availableBytes': available * gib},
        'accelerator': {'kind': 'cuda', 'totalBytes': 16 * gib, 'freeBytes': 16 * gib}}
    if refused:
        with pytest.raises(ValueError, match='actual free memory'):
            assert_next_owner_capacity(SimpleNamespace(), owner, hardware=hardware)
    else:
        assert_next_owner_capacity(SimpleNamespace(), owner, hardware=hardware)


def _exact_public_t2i_storage_observation():
    """Immutable public metadata only; never downloads or materializes weights."""
    import hashlib
    import json
    from pathlib import Path
    from modiff.transformers_image_text_resources import nf4_parameter_storage
    fixture = json.loads((Path(__file__).parent / 'fixtures/flux3-pup-nf4-storage.v1.json').read_bytes())
    config_raw = fixture['configUtf8'].encode()
    header_raw = fixture['weightHeaderUtf8'].encode()
    assert hashlib.sha256(config_raw).hexdigest() == fixture['source']['configSha256']
    assert hashlib.sha256(header_raw).hexdigest() == fixture['source']['weightHeaderSha256']
    return {'artifact': {'repository': fixture['source']['repository'], 'revision': fixture['source']['revision'],
                        'weightSha256': fixture['source']['weightSha256']},
            'config': json.loads(config_raw), 'configSha256': fixture['source']['configSha256'],
            'weightHeaderSha256': fixture['source']['weightHeaderSha256'],
            **nf4_parameter_storage(json.loads(header_raw))}


def test_exact_source_reviewed_t2i_envelope_uses_complete_named_phase_budget():
    from modiff.transformers_image_text_resources import reviewed_image_text_auto_envelope
    observed = _exact_public_t2i_storage_observation()
    result = reviewed_image_text_auto_envelope(observed,
        input_tokens=4096, image_pixels=1048576, references=0, new_tokens=3584)
    assert result['requirements'] == {'systemRamBytes': 2311061504, 'vramBytes': 12127272928, 'diskFreeBytes': 0}
    assert result['cpuPreparationRequirements'] == {'systemRamBytes': 268435456, 'vramBytes': 0, 'diskFreeBytes': 0}
    assert result['inferencePeakMeasured'] is False


@pytest.mark.parametrize('available_gib,refused', [(4, True), (5, False)])
def test_actual_preview_lease_checks_cpu_ram_before_provider_preparation(tmp_path, monkeypatch, available_gib, refused):
    """Controlled internal plan/capacity, actual leased owner boundary; no fit proof."""
    import asyncio
    import modules
    from modiff import auto_resource
    from modiff.ancillary_owner_lifecycle import owner_scope, bind_preview_auto_plan, before_node
    from modiff.workflow_auto_resource import workflow_graph_hash
    import test_ancillary_execution as ancillary
    import test_ancillary_owner_lifecycle as lifecycle
    graph, _parent = ancillary.preview_graph(modules.MODULE_MAP)
    graph['nodes']['provider']['params'].update(device={'value': 'cuda:0'},
        dtype={'value': 'bfloat16'}, quantization_mode={'value': 'bnb_4bit'})
    graph['nodes']['proposal']['params'].update(max_input_tokens={'value': 4096},
        enhancement_max_new_tokens={'value': 3584})
    graph_hash = workflow_graph_hash(graph)
    graph['runtimeHints'].update(resourceMode='auto', workflowAutoPlan={'schemaVersion': 1, 'graphHash': graph_hash})
    app = lifecycle.app_for(tmp_path, monkeypatch, graph)
    app._release_runtime_caches_for_retry = lambda: pytest.fail('No global release permission')
    hardware = {'systemMemory': {'totalBytes': 32 * 1024**3, 'availableBytes': available_gib * 1024**3},
        'accelerator': {'kind': 'cuda', 'totalBytes': 16 * 1024**3, 'freeBytes': 1024**3}}
    monkeypatch.setattr(auto_resource, '_hardware_snapshot', lambda *_args: hardware)
    owner = {'nodeId': 'provider', 'modelType': 'ImageTextToText', 'consumers': ['proposal'],
        'requirements': {'systemRamBytes': 2311061504, 'vramBytes': 12127272928, 'diskFreeBytes': 0},
        'capacityRequirements': {}, 'workingMemoryPolicy': 'runtime_headroom_policy',
        'parameterResidencyRequirements': {'systemRamBytes': 0, 'vramBytes': 8552677344},
        'parameterLoadingRequirements': {'systemRamBytes': 2034237440, 'vramBytes': 2034237440},
        'workingDemandEvidence': {'kind': 'reviewed_bounded_image_text_storage',
            'cpuPreparationRequirements': {'systemRamBytes': 268435456, 'vramBytes': 0, 'diskFreeBytes': 0}},
        'settings': {'device': 'cuda:0', 'dtype': 'bfloat16', 'quantizationMode': 'bnb_4bit', 'offloadMode': 'none'}}
    plan = {'graphHash': graph_hash, 'plannedGraphHash': graph_hash, 'canAutoRun': True,
        'patches': [], 'schedule': None, 'requiresPreparation': False, 'loaders': [owner]}
    seen = []
    def callback():
        with owner_scope(app, graph, graph['runtimeHints']['executionPurpose']):
            bind_preview_auto_plan(app, plan)
            before_node(app, 'provider')
            seen.append('processor-construction-would-begin')
    async def run():
        app.loop = asyncio.get_running_loop()
        return await app._with_node_cache_lease(lambda: app._run_executor_callback(callback, serialize_model_io=True))
    try:
        if refused:
            with pytest.raises(ValueError, match='(?i)processor.*RAM|RAM.*processor'):
                asyncio.run(run())
            assert seen == []
        else:
            asyncio.run(run())
            assert seen == ['processor-construction-would-begin']
    finally:
        if app._model_executor is not None:
            app._model_executor.shutdown(wait=True)
            app._model_executor = None


@pytest.mark.parametrize('field,value', [
    ('configSha256', '0' * 64), ('weightHeaderSha256', '0' * 64),
    ('parameterStorageBytes', 1), ('quantizedLinearCount', 247),
    ('largestSourceTensorBytes', 1), ('artifact', {'repository': 'foreign/model'}),
])
def test_finite_t2i_envelope_refuses_nonmatching_observation(field, value):
    from modiff.transformers_image_text_resources import reviewed_image_text_auto_envelope
    observed = _exact_public_t2i_storage_observation()
    observed[field] = value
    with pytest.raises(ValueError, match='exact immutable'):
        reviewed_image_text_auto_envelope(observed, input_tokens=4096,
            image_pixels=1048576, references=0, new_tokens=3584)


def test_finite_t2i_envelope_refuses_config_payload_even_with_old_hash_claim():
    from modiff.transformers_image_text_resources import reviewed_image_text_auto_envelope
    observed = _exact_public_t2i_storage_observation()
    observed['config']['text_config']['hidden_size'] += 1
    with pytest.raises(ValueError, match='exact immutable'):
        reviewed_image_text_auto_envelope(observed, input_tokens=4096,
            image_pixels=1048576, references=0, new_tokens=3584)


@pytest.mark.parametrize('changed', [
    {'references': 1}, {'references': True}, {'input_tokens': True},
    {'input_tokens': 4097}, {'new_tokens': 3585}, {'new_tokens': False},
])
def test_finite_t2i_structure_refuses_before_any_artifact_observation(changed):
    from modiff.transformers_image_text_resources import reviewed_image_text_auto_envelope
    values = {'input_tokens': 4096, 'image_pixels': 1048576, 'references': 0, 'new_tokens': 3584, **changed}
    with pytest.raises(ValueError, match='T2I.only'):
        reviewed_image_text_auto_envelope(None, **values)


@pytest.mark.parametrize('changed', [
    {'operation': None}, {'operation': 'generate'}, {'do_sample': True}, {'do_sample': 0},
    {'num_beams': True}, {'num_beams': 2}, {'repetition_penalty': '1.0'},
    {'use_chat_template': None}, {'use_chat_template': False}, {'layout_conditioning': {}},
    {'images': []}, {'video': []}, {'min_new_tokens': True}, {'min_new_tokens': 3585},
])
def test_t2i_auto_request_controls_do_not_coerce_or_change_generation(changed):
    from modiff.transformers_image_text_resources import validate_image_text_t2i_request
    with pytest.raises(ValueError, match='Image/text Auto'):
        validate_image_text_t2i_request({'operation': 'enhance_prompt', **changed}, references=0)


def test_native_reference_callable_guard_uses_real_official_fallbacks(monkeypatch, tmp_path):
    """Actual official Python callables, no neural factory/CUDA/fit proof."""
    import hashlib
    import importlib
    from pathlib import Path
    from modiff import transformers_image_text_resources as resources
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    names = ('transformers.models.qwen3_5.modeling_qwen3_5',
             'transformers.integrations.hub_kernels', 'transformers.integrations.sdpa_attention',
             'transformers.generation.utils')
    sources = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__)
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == resources._REVIEWED_NATIVE_SOURCES[name]
        sources[name] = (module, path, body, None)
    resources._require_reference_functions(dict(sources))
    # Qwen3.5 inherits this exact official method; no fabricated Qwen override.
    resources._require_native_function(qwen.Qwen3_5ForConditionalGeneration.prepare_inputs_for_generation,
        dict(sources), leaf_module='transformers.generation.utils',
        leaf_qualname='GenerationMixin.prepare_inputs_for_generation')
    with pytest.raises(ValueError, match='declared implementation'):
        resources._require_native_function(qwen.torch_recurrent_gated_delta_rule, dict(sources),
            leaf_module=qwen.__name__, leaf_qualname='torch_chunk_gated_delta_rule')
    with monkeypatch.context() as control:
        control.setattr(qwen, 'torch_chunk_gated_delta_rule', lambda *_args, **_kwargs: None)
        with pytest.raises(ValueError, match='native|callable'):
            resources._require_reference_functions(dict(sources))
    function = qwen.torch_chunk_gated_delta_rule
    cells = dict(zip(function.__code__.co_freevars, function.__closure__))
    original = cells['implementation'].cell_contents
    try:
        cells['implementation'].cell_contents = lambda *_args, **_kwargs: None
        with pytest.raises(ValueError, match='alternate implementation'):
            resources._require_reference_functions(dict(sources))
    finally:
        cells['implementation'].cell_contents = original
    resources._require_reference_functions(dict(sources))
    with monkeypatch.context() as control:
        control.setitem(qwen.ALL_ATTENTION_FUNCTIONS._global_mapping, 'sdpa', lambda *_args: None)
        with pytest.raises(ValueError, match='SDPA dispatch'):
            resources._require_reference_functions(dict(sources))
    # Change only a private copied source path; never installed library bytes.
    path = tmp_path / 'modeling_qwen3_5.py'
    path.write_bytes(sources[names[0]][2] + b'\n# changed native source\n')
    with monkeypatch.context() as control:
        control.setattr(qwen, '__file__', str(path))
        # Isolate this one-file tamper negative; this is not a runtime witness.
        control.setattr(resources, '_REVIEWED_NATIVE_SOURCES',
            {names[0]: resources._REVIEWED_NATIVE_SOURCES[names[0]]})
        with pytest.raises(ValueError, match='source changed'):
            resources._checked_native_sources()


def test_actual_generation_configuration_guards_reject_histories_and_alternate_cache(actual_pup, monkeypatch):
    """Actual tiny official module/config; neither full recipe nor fit proof."""
    from modiff.transformers_image_text_resources import validate_image_text_t2i_generation_config
    model = actual_pup[0]['model']
    # The existing tiny fixture deliberately selects eager. Prove refusal,
    # then use the official API to select SDPA for this policy-only control.
    with pytest.raises(ValueError, match='generation/cache/attention'):
        validate_image_text_t2i_generation_config(model)
    model.set_attn_implementation('sdpa')
    assert validate_image_text_t2i_generation_config(model) is True
    for field, value in [('use_cache', False), ('output_scores', True),
                         ('output_logits', True), ('output_attentions', True),
                         ('output_hidden_states', True), ('return_dict_in_generate', True),
                         ('cache_implementation', 'static'), ('prefill_chunk_size', 128),
                         ('num_return_sequences', 2), ('prompt_lookup_num_tokens', 4),
                         ('compile_config', object()), ('use_mtp', True),
                         ('penalty_alpha', 0.6), ('output_scores', 0),
                         ('num_return_sequences', True), ('num_beams', True)]:
        with monkeypatch.context() as control:
            control.setattr(model.generation_config, field, value)
            with pytest.raises(ValueError, match='generation/cache/attention'):
                validate_image_text_t2i_generation_config(model)
    with monkeypatch.context() as control:
        control.setitem(vars(model), 'forward', lambda *_args, **_kwargs: None)
        with pytest.raises(ValueError, match='replaced|hooked'):
            validate_image_text_t2i_generation_config(model)
    assert validate_image_text_t2i_generation_config(model) is True



def _controlled_t2i_storage_case():
    """CPU structural fields, not a native BnB object or Auto/CUDA proof."""
    from types import SimpleNamespace
    import torch
    weight = torch.nn.Parameter(torch.zeros((4, 1), dtype=torch.uint8), requires_grad=False)
    nested = SimpleNamespace(absmax=torch.ones(1, dtype=torch.float32),
        code=torch.arange(256, dtype=torch.float32), dtype=torch.float32,
        blocksize=256, shape=None, quant_type=None, offset=None, state2=None, nested=False)
    weight.quant_state = SimpleNamespace(absmax=torch.ones(1, dtype=torch.uint8),
        code=torch.arange(16, dtype=torch.float32), offset=torch.zeros((), dtype=torch.float32),
        dtype=torch.bfloat16, blocksize=64, shape=(2, 4), state2=nested,
        nested=True, quant_type='nf4')
    name = 'model.language_model.layers.0.mlp.up_proj.weight'
    module = SimpleNamespace(weight=weight, out_features=2, in_features=4)
    parameters = {'embedding.weight': torch.nn.Parameter(torch.zeros((8, 4), dtype=torch.bfloat16)),
                  name: weight}
    buffers = {'rope.inv_freq': torch.ones(2, dtype=torch.float32)}
    metadata = {'embedding.weight': {'shape': [8, 4], 'dtype': 'BF16'},
                name: {'shape': [2, 4], 'dtype': 'BF16'}}
    return parameters, buffers, metadata, {name: module}, {name: [2, 4]}, {'rope.inv_freq': [2]}, torch.device('cpu'), 8192


@pytest.mark.parametrize('mutation', ['none', 'dense-dtype', 'dense-shape', 'dense-view',
    'extra-parameter', 'extra-buffer', 'missing-buffer', 'buffer-dtype', 'buffer-view',
    'packed-view', 'scale-dtype', 'scale-view', 'nested-scale-dtype', 'nested-scale-view',
    'code-dtype', 'nested-code-dtype', 'offset-dtype', 'code-shape', 'state-extra', 'budget',
    'parameter-device', 'buffer-device', 'scale-device', 'nested-scale-device',
    'code-device', 'nested-code-device', 'offset-device', 'packed-shape',
    'nested-code-view', 'offset-view', 'nested-state-extra', 'nested-quant-type'])
def test_t2i_storage_predicate_closes_actual_tensor_layout_gaps(mutation):
    """Actual CPU tensors exercise the production predicate, no native/fit grant."""
    import torch
    from modiff import transformers_image_text_resources as resources
    args = list(_controlled_t2i_storage_case())
    parameters, buffers, metadata, converted, expected, derived, device, budget = args
    weight = next(iter(converted.values())).weight
    state = weight.quant_state
    if mutation == 'dense-dtype':
        parameters['embedding.weight'] = torch.nn.Parameter(parameters['embedding.weight'].float())
    elif mutation == 'dense-shape':
        parameters['embedding.weight'] = torch.nn.Parameter(torch.zeros((4, 8), dtype=torch.bfloat16))
    elif mutation == 'dense-view':
        parameters['embedding.weight'] = torch.nn.Parameter(torch.zeros((9, 4), dtype=torch.bfloat16)[:8])
    elif mutation == 'extra-parameter':
        parameters['unexpected'] = torch.nn.Parameter(torch.ones(1))
    elif mutation == 'extra-buffer':
        buffers['unexpected'] = torch.ones(1)
    elif mutation == 'missing-buffer':
        buffers.clear()
    elif mutation == 'buffer-dtype':
        buffers['rope.inv_freq'] = buffers['rope.inv_freq'].double()
    elif mutation == 'buffer-view':
        buffers['rope.inv_freq'] = torch.ones(3)[:2]
    elif mutation == 'packed-view':
        weight.data = torch.zeros((5, 1), dtype=torch.uint8)[:4]
    elif mutation == 'scale-dtype':
        state.absmax = state.absmax.to(torch.int8)
    elif mutation == 'scale-view':
        state.absmax = torch.ones(2, dtype=torch.uint8)[:1]
    elif mutation == 'nested-scale-dtype':
        state.state2.absmax = state.state2.absmax.double()
    elif mutation == 'nested-scale-view':
        state.state2.absmax = torch.ones(2)[:1]
    elif mutation == 'code-dtype':
        state.code = state.code.double()
    elif mutation == 'nested-code-dtype':
        state.state2.code = state.state2.code.double()
    elif mutation == 'offset-dtype':
        state.offset = state.offset.double()
    elif mutation == 'code-shape':
        state.code = state.code.reshape(4, 4)
    elif mutation == 'state-extra':
        state.unreviewed_tensor = torch.ones(1)
    elif mutation == 'budget':
        args[-1] = 1
    elif mutation == 'parameter-device':
        parameters['embedding.weight'] = torch.nn.Parameter(parameters['embedding.weight'].to('meta'))
    elif mutation == 'buffer-device':
        buffers['rope.inv_freq'] = buffers['rope.inv_freq'].to('meta')
    elif mutation == 'scale-device':
        state.absmax = state.absmax.to('meta')
    elif mutation == 'nested-scale-device':
        state.state2.absmax = state.state2.absmax.to('meta')
    elif mutation == 'code-device':
        state.code = state.code.to('meta')
    elif mutation == 'nested-code-device':
        state.state2.code = state.state2.code.to('meta')
    elif mutation == 'offset-device':
        state.offset = state.offset.to('meta')
    elif mutation == 'packed-shape':
        weight.data = weight.data.reshape(2, 2)
    elif mutation == 'nested-code-view':
        state.state2.code = torch.zeros(257)[:256]
    elif mutation == 'offset-view':
        state.offset = torch.zeros(2)[0]
    elif mutation == 'nested-state-extra':
        state.state2.unreviewed_tensor = torch.ones(1)
    elif mutation == 'nested-quant-type':
        state.state2.quant_type = 'nf4'
    if mutation == 'none':
        resources._validate_t2i_tensor_storage(*args)
    else:
        with pytest.raises(ValueError, match='storage|census|tensor|QuantState'):
            resources._validate_t2i_tensor_storage(*args)


@pytest.mark.parametrize('class_name', ['none', 'Qwen3_5Model', 'Qwen3_5TextModel'])
def test_t2i_actual_inner_native_forward_bindings(actual_pup, monkeypatch, class_name):
    """Actual official CPU classes/code; does not bypass exact CUDA entry guards."""
    import importlib
    from pathlib import Path
    from modiff import transformers_image_text_resources as resources
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    names = ('transformers.models.qwen3_5.modeling_qwen3_5',
        'transformers.generation.utils', 'transformers.generation.configuration_utils',
        'torch.utils._contextlib', 'transformers.utils.generic',
        'transformers.integrations.hub_kernels', 'transformers.integrations.accelerate',
        'transformers.utils.output_capturing')
    sources = {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__)
        sources[name] = (module, path, path.read_bytes(), None)
    model = actual_pup[0]['model']
    resources._require_t2i_generation_functions(model, dict(sources))
    if class_name != 'none':
        with monkeypatch.context() as control:
            control.setattr(getattr(qwen, class_name), 'forward', lambda *_args, **_kwargs: None)
            with pytest.raises(ValueError, match='native|callable|declared'):
                resources._require_t2i_generation_functions(model, dict(sources))
    resources._require_t2i_generation_functions(model, dict(sources))


def test_t2i_publisher_fp32_dense_floor_and_native_derived_buffer_dimensions():
    """Reviewed header can retain FP32 or safely downcast it, not promote BF16."""
    import torch
    from modiff import transformers_image_text_resources as resources
    args = list(_controlled_t2i_storage_case())
    args[2]['embedding.weight']['dtype'] = 'F32'
    resources._validate_t2i_tensor_storage(*args)
    args[0]['embedding.weight'] = torch.nn.Parameter(args[0]['embedding.weight'].float())
    resources._validate_t2i_tensor_storage(*args)
    config = _exact_public_t2i_storage_observation()['config']
    expected = {
        'model.language_model.rotary_emb.inv_freq': [32],
        'model.language_model.rotary_emb.original_inv_freq': [32],
        'model.visual.rotary_pos_emb.inv_freq': [18],
        'model.visual.rotary_pos_emb.original_inv_freq': [18],
    }
    assert resources._t2i_derived_buffer_shapes(config) == expected
    assert sum(shape[0] * 4 for shape in expected.values()) == 400



def _intercept_metadata_read(monkeypatch, target, mutation, *, enabled=lambda: True):
    """Small stream sentinel simulates growth; never allocates an oversized file."""
    import os
    from pathlib import Path
    original_open = Path.open
    original_body = target.read_bytes()
    seen = []
    class UnboundedReadAttempt(RuntimeError):
        pass
    class Reader:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            self.handle.__enter__()
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def __getattr__(self, name):
            return getattr(self.handle, name)
        def read(self, size=-1):
            seen.append(size)
            if mutation == 'growth':
                if size < 0:
                    raise UnboundedReadAttempt('A post-validation growth would trigger an unbounded allocation.')
                return (original_body + b' ')[:size]
            if mutation == 'short-read':
                return original_body[:-1] if size < 0 else original_body[:-1][:size]
            result = self.handle.read(size)
            if mutation == 'identity-change':
                current = target.stat()
                os.utime(target, ns=(current.st_atime_ns, current.st_mtime_ns + 10_000_000))
            return result
    def opened(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if Path(path).absolute() == target.absolute() and enabled():
            return Reader(handle)
        return handle
    monkeypatch.setattr(Path, 'open', opened)
    return seen, len(original_body)


@pytest.mark.parametrize('mutation', ['none', 'growth', 'short-read', 'identity-change'])
def test_resource_config_read_is_bounded_after_real_cache_admission(actual_pup, monkeypatch, mutation):
    """Actual tiny fixture cache admission; no large files or eligibility bypass."""
    from pathlib import Path
    from modiff import prompt_enhancement as enhancement
    from modiff.config import CONFIG
    from modiff.transformers_image_text_resources import inspect_image_text_nf4_storage
    target = Path(CONFIG.hf['cache_dir']) / ('models--' + pup.REPOSITORY.replace('/', '--')) / 'snapshots' / pup.REVISION / 'config.json'
    admitted = False
    original_admission = enhancement.validate_cached_selection
    def admission(*args, **kwargs):
        nonlocal admitted
        value = original_admission(*args, **kwargs)
        admitted = True
        return value
    monkeypatch.setattr(enhancement, 'validate_cached_selection', admission)
    seen, original_size = _intercept_metadata_read(monkeypatch, target, mutation, enabled=lambda: admitted)
    if mutation == 'none':
        # Existing fixture deliberately pairs publisher config with tiny
        # weights. Unchanged config reaches its genuine later shape refusal.
        with pytest.raises(ValueError, match='targets/shapes differ'):
            inspect_image_text_nf4_storage({'source': 'hub', 'value': pup.REPOSITORY}, pup.REVISION)
    else:
        with pytest.raises(ValueError, match='configuration.*changed|configuration.*bound|declaration.*bounded'):
            inspect_image_text_nf4_storage({'source': 'hub', 'value': pup.REPOSITORY}, pup.REVISION)
    assert admitted and seen
    if mutation == 'growth':
        assert all(0 < size <= 2 * 1024**2 + 1 for size in seen)


@pytest.mark.parametrize('mutation', ['none', 'growth', 'short-read', 'identity-change'])
def test_native_source_read_is_bounded_after_size_validation(tmp_path, monkeypatch, mutation):
    """Identical private copy of official source, not a full runtime witness."""
    import hashlib
    from pathlib import Path
    from modiff import transformers_image_text_resources as resources
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    body = Path(qwen.__file__).read_bytes()
    assert hashlib.sha256(body).hexdigest() == resources._REVIEWED_NATIVE_SOURCES[qwen.__name__]
    target = tmp_path / 'modeling_qwen3_5.py'
    target.write_bytes(body)
    monkeypatch.setattr(qwen, '__file__', str(target))
    monkeypatch.setattr(resources, '_REVIEWED_NATIVE_SOURCES',
        {qwen.__name__: resources._REVIEWED_NATIVE_SOURCES[qwen.__name__]})
    seen, original_size = _intercept_metadata_read(monkeypatch, target, mutation)
    if mutation == 'none':
        result = resources._checked_native_sources()
        assert result[qwen.__name__][2] == body
    else:
        with pytest.raises(ValueError, match='source changed|finite boundary'):
            resources._checked_native_sources()
    assert seen
    if mutation == 'growth':
        assert all(0 < size <= original_size + 1 for size in seen)



@pytest.mark.parametrize('case', [
    'native-copy', 'native-base', 'foreign-bound-owner', 'unbound-copy',
    'foreign-wrapped-owner', 'foreign-wrapped-function', 'plain-wrapped-function',
    'foreign-copy', 'copy-defaults', 'copy-kwdefaults', 'shared-defaults',
    'copy-signature', 'shared-signature', 'base-descriptor', 'missing-metadata',
    'base-wrapped', 'cycle', 'wrong-leaf'])
def test_native_auto_factory_documentation_copy_is_exact_and_tamper_closed(monkeypatch, case):
    """Actual TF5.18 methods/code only; no from_pretrained invocation or runtime claim."""
    import hashlib
    import inspect
    from pathlib import Path
    from types import FunctionType, MethodType
    from transformers.models.auto import auto_factory, modeling_auto
    from modiff import transformers_image_text_resources as resources
    owner = modeling_auto.AutoModelForImageTextToText
    base = auto_factory._BaseAutoModelClass
    copied = vars(owner)['from_pretrained'].__func__
    base_function = vars(base)['from_pretrained'].__func__
    method = owner.from_pretrained
    qualname = '_BaseAutoModelClass.from_pretrained'
    path = Path(auto_factory.__file__)
    body = path.read_bytes()
    assert hashlib.sha256(body).hexdigest() == resources._REVIEWED_NATIVE_SOURCES[auto_factory.__name__]
    sources = {auto_factory.__name__: (auto_factory, path, body, None)}
    # Source-generated copy_func metadata is a bound base classmethod, while
    # the copied body itself is the same native code and has no closure.
    assert type(vars(owner)['from_pretrained']) is classmethod
    assert copied.__code__ is base_function.__code__
    assert type(copied.__wrapped__) is MethodType
    assert copied.__wrapped__.__self__ is base
    assert copied.__wrapped__.__func__ is base_function
    assert copied.__defaults__ is copied.__kwdefaults__ is copied.__closure__ is None
    if case == 'native-base':
        method = base.from_pretrained
    elif case == 'foreign-bound-owner':
        foreign = type('UnregisteredAutoOwner', (owner,), {})
        method = foreign.from_pretrained
    elif case == 'unbound-copy':
        method = copied
    elif case == 'foreign-wrapped-owner':
        monkeypatch.setattr(copied, '__wrapped__', MethodType(base_function, object()))
    elif case == 'foreign-wrapped-function':
        monkeypatch.setattr(copied, '__wrapped__', MethodType(lambda *_args, **_kwargs: None, base))
    elif case == 'plain-wrapped-function':
        monkeypatch.setattr(copied, '__wrapped__', base_function)
    elif case == 'foreign-copy':
        foreign = FunctionType(copied.__code__, copied.__globals__, copied.__name__)
        foreign.__wrapped__ = base.from_pretrained
        method = MethodType(foreign, owner)
    elif case == 'copy-defaults':
        monkeypatch.setattr(copied, '__defaults__', ('unreviewed',))
    elif case == 'copy-kwdefaults':
        monkeypatch.setattr(copied, '__kwdefaults__', {'unreviewed': True})
    elif case == 'shared-defaults':
        defaults = ('unreviewed',)
        monkeypatch.setattr(copied, '__defaults__', defaults)
        monkeypatch.setattr(base_function, '__defaults__', defaults)
    elif case == 'copy-signature':
        monkeypatch.setattr(copied, '__signature__', inspect.Signature(), raising=False)
    elif case == 'shared-signature':
        signature = inspect.Signature()
        monkeypatch.setattr(copied, '__signature__', signature, raising=False)
        monkeypatch.setattr(base_function, '__signature__', signature, raising=False)
    elif case == 'base-descriptor':
        monkeypatch.setattr(base, 'from_pretrained', classmethod(lambda *_args, **_kwargs: None))
    elif case == 'missing-metadata':
        monkeypatch.delattr(copied, '__wrapped__')
    elif case == 'base-wrapped':
        monkeypatch.setattr(base_function, '__wrapped__', lambda *_args, **_kwargs: None, raising=False)
    elif case == 'cycle':
        monkeypatch.setattr(copied, '__wrapped__', copied)
    elif case == 'wrong-leaf':
        qualname = '_BaseAutoModelClass.from_config'
    if case in {'native-copy', 'native-base'}:
        resources._require_native_function(method, sources,
            leaf_module=auto_factory.__name__, leaf_qualname=qualname)
    else:
        with pytest.raises(ValueError, match='native|callable|copied|factory|declared'):
            resources._require_native_function(method, sources,
                leaf_module=auto_factory.__name__, leaf_qualname=qualname)
