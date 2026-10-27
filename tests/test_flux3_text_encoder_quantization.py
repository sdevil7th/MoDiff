"""Literal producer/owner boundaries; synthetic contracts are not native fit."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from modiff import flux3_text_encoder_quantization as policy


def graph(model='Flux3DistilledModularPipeline'):
    repository, revision = policy.PARENTS[model]
    values = {'model_type': model, 'repo_id': {'source': 'hub', 'value': repository}, 'revision': revision,
              'dtype': 'bfloat16', 'device': 'cuda:0', 'weight_format': 'nvfp4', 'weight_loading_mode': 'cpu',
              'offload_mode': 'model_cpu', 'auto_offload': True}
    quantization = {'model_id': {'source': 'hub', 'value': repository}, 'subfolder': 'text_encoder',
        'component': 'text_encoder', 'quant_type': 'bnb_4bit', 'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_use_double_quant': True,
        'llm_int8_threshold': 6.0, 'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': [],
        'skip_module_names': 'model.visual,lm_head'}
    return {'models': {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader',
                      'params': {**{key: {'value': value} for key, value in values.items()},
                                 'quant_config': {'sourceId': 'quant', 'sourceKey': 'quantization_config'}}},
            'quant': {'module': 'modules.ModularDiffusers', 'action': 'QuantizationConfigNode',
                      'params': {key: {'value': value} for key, value in quantization.items()}}}


@pytest.mark.parametrize('model', policy.PARENTS)
def test_exact_literal_producer_is_inspected_without_execution_or_mutation(model, monkeypatch):
    import builtins
    nodes = graph(model)
    before = deepcopy(nodes)
    original = builtins.__import__
    def bounded_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'diffusers', 'transformers', 'bitsandbytes'}:
            raise AssertionError('The pure graph inspector imported a native runtime.')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', bounded_import)
    descriptor = policy.inspect_text_encoder_nf4(nodes, 'models')
    assert policy.validate_text_encoder_nf4_descriptor(descriptor) is True
    assert descriptor['producerId'] == 'quant'
    assert descriptor['skipModules'] == ['model.visual', 'lm_head']
    assert descriptor['loadDevice'] == 'cuda:0'
    assert nodes == before


@pytest.mark.parametrize('field,value', [
    ('model_id', {'source': 'hub', 'value': 'mutable/other'}), ('component', 'transformer'),
    ('subfolder', 'transformer'), ('quant_type', 'bnb_8bit'), ('bnb_4bit_quant_type', 'fp4'),
    ('bnb_4bit_compute_dtype', 'float16'), ('bnb_4bit_use_double_quant', False),
    ('bnb_4bit_use_double_quant', 1), ('llm_int8_skip_modules', [0]),
    ('skip_module_names', ''), ('skip_module_names', 'model.visual'),
    ('skip_module_names', 'lm_head,model.visual'), ('llm_int8_threshold', True),
    ('llm_int8_has_fp16_weight', True), ('unknown_override', True),
])
def test_changed_or_undeclared_producer_controls_are_refused(field, value):
    nodes = graph()
    nodes['quant']['params'][field] = {'value': value}
    with pytest.raises(ValueError):
        policy.inspect_text_encoder_nf4(nodes, 'models')


@pytest.mark.parametrize('field,value', [
    ('weight_format', 'fp8r'), ('weight_format', 'native'), ('dtype', 'float16'), ('device', 'cpu'),
    ('device', 'cuda:1'), ('offload_mode', 'none'), ('auto_offload', False),
    ('weight_loading_mode', 'cuda'), ('trust_remote_code', True), ('lora_list', [{}]),
    ('unet', {}), ('vae', {}), ('controlnet', {}), ('revision', 'main'),
])
def test_unreviewed_parent_or_placement_never_borrows_nf4(field, value):
    nodes = graph()
    nodes['models']['params'][field] = {'value': value}
    with pytest.raises(ValueError):
        policy.inspect_text_encoder_nf4(nodes, 'models')


@pytest.mark.parametrize('where', ['owner', 'producer', 'custom', 'chained', 'wrong-port'])
def test_connected_custom_and_chained_supplier_boundaries(where):
    nodes = graph()
    if where == 'owner':
        nodes['models']['params']['dtype'] = {'sourceId': 'unknown', 'sourceKey': 'output'}
    elif where == 'producer':
        nodes['quant']['params']['bnb_4bit_compute_dtype'] = {'sourceId': 'unknown', 'sourceKey': 'output'}
    elif where == 'custom':
        nodes['quant']['module'] = 'custom.attacker'
    elif where == 'chained':
        nodes['quant']['params']['config'] = {'sourceId': 'other', 'sourceKey': 'quantization_config'}
    else:
        nodes['models']['params']['quant_config']['sourceKey'] = 'config_info'
    with pytest.raises(ValueError):
        policy.inspect_text_encoder_nf4(nodes, 'models')


def test_no_quantization_preserves_existing_native_defaults():
    nodes = graph()
    nodes['models']['params']['quant_config'] = {'value': None}
    nodes['models']['params']['weight_format'] = {'value': 'fp8r'}
    assert policy.inspect_text_encoder_nf4(nodes, 'models') is None


@pytest.mark.parametrize('where,value', [('nodes', []), ('owner', None), ('owner-params', []),
    ('producer', []), ('producer-params', None)])
def test_malformed_source_containers_refuse_as_value_errors(where, value):
    nodes = graph()
    if where == 'nodes':
        nodes = value
    elif where == 'owner':
        nodes['models'] = value
    elif where == 'owner-params':
        nodes['models']['params'] = value
    elif where == 'producer':
        nodes['quant'] = value
    else:
        nodes['quant']['params'] = value
    with pytest.raises(ValueError):
        policy.inspect_text_encoder_nf4(nodes, 'models')


@pytest.mark.parametrize('field,value', [('producerId', 'other'), ('signature', '0' * 64),
    ('doubleQuant', 1), ('skipModules', ['model.visual']), ('loadDevice', 'cpu'), ('extra', None)])
def test_descriptor_cannot_be_modified_or_extended(field, value):
    descriptor = policy.inspect_text_encoder_nf4(graph(), 'models')
    descriptor[field] = value
    with pytest.raises(ValueError):
        policy.validate_text_encoder_nf4_descriptor(descriptor)


@pytest.mark.parametrize('value', ['a..b', '__private.value', 'model.__dict__', 'a,a', 'a,', 'a[0]', 'a/b', 1])
def test_generic_explicit_module_paths_have_a_finite_literal_grammar(value):
    with pytest.raises(ValueError):
        policy.parse_module_skip_names(value)


def test_explicit_names_do_not_depend_on_cached_layer_indices():
    assert policy.parse_module_skip_names(' model.visual, lm_head ') == ['model.visual', 'lm_head']
    assert policy.parse_module_skip_names('') == []


def test_language_census_preserves_vision_embeddings_and_tied_output():
    shapes = policy.expected_language_linear_shapes()
    assert len(shapes) == 252
    assert sum(rows * columns for rows, columns in shapes.values()) == 3_633_315_840
    assert shapes['model.language_model.layers.0.self_attn.q_proj'] == (4096, 2560)
    assert not any(name.startswith('model.visual') or name in {'lm_head', 'model.language_model.embed_tokens'} for name in shapes)


def test_component_kwargs_merge_official_bfl_without_replacing_text_policy(monkeypatch, tmp_path):
    from modiff.modular_weight_formats import ResolvedWeightFormat
    module = ModuleType('diffusers')
    transformer = type('SyntheticTransformer', (), {})
    module.SyntheticTransformer = transformer
    module.BFLQuantizationConfig = lambda **kwargs: ('official-bfl', kwargs)
    monkeypatch.setitem(sys.modules, 'diffusers', module)
    monkeypatch.setattr(ResolvedWeightFormat, 'assert_current', lambda self: None)
    descriptor = policy.inspect_text_encoder_nf4(graph(), 'models')
    text_config = object()
    mapping = {'text_encoder': text_config}
    monkeypatch.setattr(policy, 'validate_text_encoder_nf4_config', lambda actual, expected: actual is mapping)
    monkeypatch.setattr(policy, 'text_encoder_nf4_loader_kwargs', lambda value:
                        {'device_map': {'text_encoder': {'': 'cuda:0'}}})
    recipe = {'loaderClass': 'SyntheticTransformer', 'repository': 'synthetic/unit', 'revision': 'a' * 40,
              'weight': {'path': 'variants/nvfp4/model.safetensors'}}
    spec = SimpleNamespace(type_hint=transformer, pretrained_model_name_or_path=recipe['repository'],
                           revision=recipe['revision'], subfolder='transformer', variant=None)
    pipeline = SimpleNamespace(_component_specs={'transformer': spec}, transformer=None)
    binding = ResolvedWeightFormat(recipe, tmp_path, Path('weight'), Path('config'), (), {}, None, descriptor)
    config, kwargs = binding.configure(pipeline, 'bfloat16', mapping)
    assert config['text_encoder'] is text_config
    assert config['transformer'] == ('official-bfl', {'compute_dtype': 'bfloat16'})
    assert kwargs['device_map'] == {'text_encoder': {'': 'cuda:0'}}
    assert spec.subfolder is None
    assert pipeline._modiff_text_encoder_quantization == descriptor


def test_combined_provider_is_required_even_when_plain_diffusers_is_ready(monkeypatch):
    from modiff import optional_runtime_execution, runtime_overlays
    calls = []
    def observed(profiles, **kwargs):
        calls.append(profiles[0].optional_runtime_profiles)
        return {'state': 'active'}
    monkeypatch.setattr(optional_runtime_execution, 'optional_runtime_requirement_for_profiles', observed)
    monkeypatch.setattr(runtime_overlays, 'required_diffusers_provider_available', lambda identity: False)
    assert policy.text_encoder_nf4_runtime_ready() is False
    monkeypatch.setattr(runtime_overlays, 'required_diffusers_provider_available', lambda identity: True)
    assert policy.text_encoder_nf4_runtime_ready() is True
    assert calls == [('huggingface-diffusers-flux3-7ecb1e1-bitsandbytes-0.50.0',)] * 2


def test_real_generic_node_preserves_explicit_skips_and_official_nf4_config():
    from modules.ModularDiffusers.loaders import QuantizationConfigNode
    from transformers import BitsAndBytesConfig
    node = SimpleNamespace(_cached_layers={'unused': {'misleading': ['foreign.weight']}})
    values = {key: field['value'] for key, field in graph()['quant']['params'].items()}
    output = QuantizationConfigNode.execute(node, **values)
    descriptor = policy.inspect_text_encoder_nf4(graph(), 'models')
    config = policy.validate_text_encoder_nf4_config(output['quantization_config'], descriptor)
    assert type(config) is BitsAndBytesConfig
    assert config.llm_int8_skip_modules == ['model.visual', 'lm_head']
    config.bnb_4bit_use_double_quant = False
    with pytest.raises(ValueError):
        policy.validate_text_encoder_nf4_config(output['quantization_config'], descriptor)
    values['llm_int8_skip_modules'] = [0]
    with pytest.raises(ValueError, match='not both'):
        QuantizationConfigNode.execute(node, **values)


def test_component_reuse_policy_preserves_full_nf4_identity_without_aliasing():
    from modules.ModularDiffusers.loaders import _component_runtime_policy
    descriptor = policy.inspect_text_encoder_nf4(graph(), 'models')
    request = {'text_encoder_quantization': descriptor}
    stored = _component_runtime_policy('text_encoder', request)
    assert stored == {'text_encoder_quantization': descriptor, 'weight_loading_device': 'cuda:0'}
    assert _component_runtime_policy('vae', request) == {}
    descriptor['skipModules'].append('foreign')
    assert stored['text_encoder_quantization']['skipModules'] == ['model.visual', 'lm_head']


def test_strict_loader_verifies_cuda_then_registers_then_verifies_cpu(monkeypatch):
    from modules.ModularDiffusers import loaders
    from transformers import Qwen3VLForConditionalGeneration
    import torch
    events = []
    descriptor = policy.inspect_text_encoder_nf4(graph(), 'models')
    repository, revision = policy.PARENTS['Flux3DistilledModularPipeline']
    component = SimpleNamespace(device='cuda:0')
    def load(**kwargs):
        events.append(('load', kwargs['device_map']))
        return component
    spec = SimpleNamespace(type_hint=Qwen3VLForConditionalGeneration,
        pretrained_model_name_or_path=repository, revision=revision, subfolder='text_encoder', variant=None, load=load)
    def register(**kwargs):
        assert kwargs == {'text_encoder': component}
        events.append(('register', component.device))
        component.device = 'cpu'
    pipeline = SimpleNamespace(_component_specs={'text_encoder': spec},
        _components_manager=SimpleNamespace(_auto_offload_enabled=True), register_components=register)
    def validate(actual, selected, *, device):
        assert actual is component and selected == descriptor and actual.device == device
        events.append(('validate', device))
        return {'currentDevice': device}
    monkeypatch.setattr(policy, 'validate_text_encoder_nf4_model', validate)
    def capacity(binding, component, *, context=None, registering=False):
        events.append(('capacity', component, registering))
        return {'checks': []}
    monkeypatch.setattr(policy, 'assert_text_encoder_nf4_loading_capacity', capacity)
    diagnostics = {}
    loaders.load_components_strict(pipeline, ['text_encoder'], required_names={'text_encoder'}, model_id=repository,
        dtype=torch.bfloat16, offload_mode='model_cpu', quant_config={}, diagnostics=diagnostics,
        component_load_kwargs={'config': {'text_encoder': object()}, 'device_map': {'text_encoder': {'': 'cuda:0'}}},
        weight_format_binding=SimpleNamespace(text_encoder_quantization=descriptor,
                                              recipe={'repository': repository, 'revision': revision}))
    assert events == [('capacity', 'text_encoder', False), ('load', {'': 'cuda:0'}), ('validate', 'cuda:0'),
                      ('capacity', 'text_encoder', True), ('register', 'cuda:0'), ('validate', 'cpu')]
    assert diagnostics['text_encoder_quantization'] == {'currentDevice': 'cpu'}


@pytest.mark.parametrize('identity', ['flux3-klein:official-provider:v1', 'flux3-klein-base:official-provider:v1'])
@pytest.mark.parametrize('task', ['text_to_image', 'edit_image'])
def test_public_source_context_binds_literal_quantization_before_dispatch(monkeypatch, identity, task):
    from test_flux3_auto_resource import flux_graph
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    from modiff.native_component_lora import ComponentLoraContext, current_component_lora_context, native_component_graph_scope
    from modules.ModularDiffusers.loaders import ModelsLoader
    source = flux_graph(identity, task)
    profile = DIFFUSERS_EXECUTION_PROFILES[identity]
    inspected = graph(profile.model_type)
    source['nodes']['models']['params'].update(inspected['models']['params'])
    source['nodes']['models']['params']['workflow_id'] = {'value': 'text2image' if task == 'text_to_image' else 'image_conditioned'}
    if 'guider' in source['nodes']:
        source['nodes']['guider']['params']['model_type'] = {'value': profile.model_type}
    source['nodes']['quant'] = inspected['quant']
    source['paths'][0].insert(0, 'quant')
    def preflight(**kwargs):
        return 'hub', kwargs['repo_id']['value'], kwargs['revision'], 'modular_model_index.json', {}
    monkeypatch.setattr(ModelsLoader, '_preflight_reviewed_builtin_selection', staticmethod(preflight))
    # This test proves source/dispatch identity. Native artifact bytes are
    # separately checked by the existing single-file integration tests.
    monkeypatch.setattr(ComponentLoraContext, 'validate_weight_format', lambda *args, **kwargs: None)
    supplied = object()
    def actual_config(value, descriptor):
        if value is not supplied:
            raise ValueError('Different actual producer result')
    monkeypatch.setattr(policy, 'validate_text_encoder_nf4_config', actual_config)
    monkeypatch.setattr(policy, 'text_encoder_nf4_runtime_ready', lambda: True)
    with native_component_graph_scope(source, 'synthetic-source-fingerprint'):
        context = current_component_lora_context('models')
        assert context.text_encoder_quantization == policy.inspect_text_encoder_nf4(source['nodes'], 'models')
        assert ('text-encoder-quantization-v1', context.text_encoder_quantization['signature']) in context.signature
        values = {key: value.get('value') for key, value in source['nodes']['models']['params'].items()}
        values['quant_config'] = supplied
        values['_reviewed_builtin_identity'] = ('hub', profile.default_repo, values['revision'], *context.index_stamp)
        context.validate_source_submission(values)
        context.validate_resolved(values)
        with pytest.raises(ValueError):
            context.validate_source_submission({**values, 'quant_config': object()})
        monkeypatch.setattr(policy, 'text_encoder_nf4_runtime_ready', lambda: False)
        with pytest.raises(ValueError, match='before owner preparation'):
            context.validate_source_submission(values)
        monkeypatch.setattr(policy, 'text_encoder_nf4_runtime_ready', lambda: True)
        source['nodes']['quant']['params']['bnb_4bit_use_double_quant']['value'] = False
        with pytest.raises(ValueError):
            context.validate_source_submission(values)


def capacity_context():
    return {'bound': {}, 'reserve': {'components': {'transformer': {'conservativeCpuOverlapBytes': 20},
            'vae': {'conservativeCpuOverlapBytes': 8}}},
        'phases': {'textSourceBytes': 30, 'textTargetBytes': 10, 'cudaQuantizationBytes': 40}, 'checks': []}


def capacity_hardware():
    return {'system': {'ram_total': 32 * 1024**3, 'ram_available': 12 * 1024**3},
            'devices': [{'device': 'cuda:0', 'backend': 'cuda', 'memory_kind': 'dedicated',
                         'planning_memory_total': 16 * 1024**3, 'planning_memory_free': 12 * 1024**3}]}


@pytest.mark.parametrize('component,ram,gpu', [('text_encoder', 30, 40), ('transformer', 20, 0), ('vae', 8, 0)])
def test_fresh_component_checks_keep_full_source_and_unchanged_headroom(monkeypatch, component, ram, gpu):
    from modiff import hardware
    calls = []
    def observe(**kwargs):
        calls.append(kwargs)
        return capacity_hardware()
    monkeypatch.setattr(hardware, 'get_hardware_snapshot', observe)
    context = policy.assert_text_encoder_nf4_loading_capacity(None, component, context=capacity_context())
    checked = context['checks'][-1]
    assert checked['ramRequiredBytes'] == ram + 4 * 1024**3
    assert checked['vramRequiredBytes'] == gpu + 2 * 1024**3
    assert calls[0]['refresh'] is True


@pytest.mark.parametrize('pool,value', [('ram_available', 0), ('planning_memory_free', 0),
    ('ram_available', None), ('planning_memory_free', None), ('planning_memory_free', -1),
    ('planning_memory_free', 17 * 1024**3), ('ram_available', 33 * 1024**3), ('ram_available', False)])
def test_fresh_unknown_zero_and_impossible_capacity_never_gain_allocator_credit(monkeypatch, pool, value):
    from modiff import hardware
    observed = capacity_hardware()
    target = observed['system'] if pool == 'ram_available' else observed['devices'][0]
    target[pool] = value
    monkeypatch.setattr(hardware, 'get_hardware_snapshot', lambda **kwargs: observed)
    with pytest.raises(ValueError):
        policy.assert_text_encoder_nf4_loading_capacity(None, 'text_encoder', context=capacity_context())


def test_fresh_registration_rechecks_target_copy_instead_of_crediting_closed_source(monkeypatch):
    from modiff import hardware
    observed = capacity_hardware()
    monkeypatch.setattr(hardware, 'get_hardware_snapshot', lambda **kwargs: deepcopy(observed))
    context = policy.assert_text_encoder_nf4_loading_capacity(None, 'text_encoder', context=capacity_context())
    # A library/allocator holding source storage remains part of physical use.
    observed['system']['ram_available'] = 4 * 1024**3 + 9
    with pytest.raises(ValueError, match='cpu_registration capacity refused'):
        policy.assert_text_encoder_nf4_loading_capacity(None, 'text_encoder', context=context, registering=True)
    assert context['checks'][-1]['ramRequiredBytes'] == 4 * 1024**3 + 10


def test_census_is_once_per_load_but_capacity_observations_are_fresh(monkeypatch):
    from modiff import flux3_auto_resource, hardware
    context = capacity_context()
    censuses, observations = [], []
    monkeypatch.setattr(flux3_auto_resource, 'selected_artifacts', lambda *args: {'complete': True})
    def census(*args):
        censuses.append(args)
        return context['bound'], context['reserve'], {}, context['phases']
    def snapshot(**kwargs):
        observations.append(kwargs)
        return capacity_hardware()
    monkeypatch.setattr(flux3_auto_resource, 'nf4_parameter_census', census)
    monkeypatch.setattr(hardware, 'get_hardware_snapshot', snapshot)
    binding = SimpleNamespace(recipe={'modelType': 'Flux3DistilledModularPipeline',
        'repository': policy.PARENTS['Flux3DistilledModularPipeline'][0],
        'revision': policy.PARENTS['Flux3DistilledModularPipeline'][1]})
    checked = policy.assert_text_encoder_nf4_loading_capacity(binding, 'transformer')
    policy.assert_text_encoder_nf4_loading_capacity(binding, 'text_encoder', context=checked)
    assert len(censuses) == 1 and len(observations) == 2


@pytest.mark.parametrize('target,field,value', [
    ('outer', 'hidden_tensor_owner', object()), ('nested', 'hidden_tensor_owner', object()),
    ('nested', 'shape', (1,)), ('nested', 'quant_type', 'nf4'), ('nested', 'offset', object()),
    ('nested', 'state2', object()), ('nested', 'nested', True), ('nested', 'nested', 0),
])
def test_quant_state_inventory_rejects_hidden_storage_and_nested_variants(target, field, value):
    # Synthetic state objects prove this finite inventory check only. Actual
    # execution passes the official bnb.functional.QuantState type explicitly.
    class NativeState:
        pass
    def state():
        result = NativeState()
        for key in ('absmax', 'shape', 'code', 'dtype', 'blocksize', 'quant_type', 'offset', 'state2'):
            setattr(result, key, None)
        result.nested = False
        return result
    outer = state()
    outer.state2 = state()
    outer.nested = True
    policy._validate_native_quant_state_structure(outer, NativeState)
    setattr(outer if target == 'outer' else outer.state2, field, value)
    with pytest.raises(ValueError, match='QuantState'):
        policy._validate_native_quant_state_structure(outer, NativeState)


def test_quant_state_inventory_rejects_foreign_native_type():
    class NativeState:
        pass
    with pytest.raises(ValueError, match='unknown type'):
        policy._validate_native_quant_state_structure(SimpleNamespace(), NativeState)


def test_missing_composite_refuses_preflight_before_owner_or_artifact_actions(monkeypatch):
    from modiff import native_component_lora, native_disk_lora
    events = []
    descriptor = policy.inspect_text_encoder_nf4(graph(), 'models')
    context = SimpleNamespace(text_encoder_quantization=descriptor, weight_format_contract=('nvfp4',))
    monkeypatch.setattr(native_component_lora, 'current_component_lora_context', lambda owner: context)
    monkeypatch.setattr(policy, 'text_encoder_nf4_runtime_ready', lambda: False)
    monkeypatch.setattr(native_disk_lora, 'graph_disk_lora_policy', lambda *args, **kwargs: events.append('owner-policy'))
    monkeypatch.setattr(native_component_lora, 'validate_retained_factor_shapes',
                        lambda *args, **kwargs: events.append('retained-owner'))
    with pytest.raises(ValueError, match='before owner preparation'):
        native_component_lora.preflight_component_graph(SimpleNamespace(node_cache={}), {'nodes': graph()})
    assert events == []


@pytest.fixture
def consumed_quantization_boundary(monkeypatch):
    """Real Transformers config; synthetic BFL/owner boundary, no model proof."""
    import torch
    from transformers import BitsAndBytesConfig
    text = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=['model.visual', 'lm_head'])
    class BFLConfig:
        compute_dtype = torch.bfloat16
    class BFLQuantizer:
        pass
    class NVFP4:
        compute_dtype = torch.bfloat16
    class FP8R:
        compute_dtype = torch.bfloat16
    modules = {name: ModuleType(name) for name in ('diffusers', 'diffusers.quantizers', 'diffusers.quantizers.bfl',
        'diffusers.quantizers.bfl.bfl_quantizer', 'diffusers.quantizers.bfl.utils')}
    modules['diffusers'].BFLQuantizationConfig = BFLConfig
    modules['diffusers.quantizers.bfl.bfl_quantizer'].BFLQuantizer = BFLQuantizer
    modules['diffusers.quantizers.bfl.utils'].BFLNVFP4Linear = NVFP4
    modules['diffusers.quantizers.bfl.utils'].BFLFP8RLinear = FP8R
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    descriptor = policy.inspect_text_encoder_nf4(graph(), 'models')
    config = {'transformer': BFLConfig(), 'text_encoder': text}
    quantizer = BFLQuantizer()
    quantizer.compute_dtype = torch.bfloat16
    quantizer.quantization_config = config['transformer']
    packed = [NVFP4()]
    transformer = SimpleNamespace(hf_quantizer=quantizer, hf_device_map={'': torch.device('cpu')},
        modules=lambda: packed, parameters=lambda: (), buffers=lambda: ())
    binding = SimpleNamespace(text_encoder_quantization=descriptor, recipe={'weightFormat': 'nvfp4'},
                              assert_current=lambda: None)
    encoder = object()
    pipeline = SimpleNamespace(transformer=transformer, text_encoder=encoder,
        _modiff_weight_format_binding=binding, _modiff_text_encoder_quantization=descriptor)
    def validated(actual, selected, *, device):
        assert actual is encoder and selected == descriptor and device == 'cpu'
        return {'convertedLinearCount': 252}
    monkeypatch.setattr(policy, 'validate_text_encoder_nf4_model', validated)
    return SimpleNamespace(config=config, pipeline=pipeline, binding=binding, packed=packed, fp8=FP8R)


def test_successful_consumed_quantization_is_flat_bounded_and_keeps_connected_origin(consumed_quantization_boundary):
    from modiff.execution_input_provenance import capture_generation_inputs
    boundary = consumed_quantization_boundary
    original_text = boundary.config['text_encoder']
    summary = policy.text_encoder_nf4_consumed_quantization(boundary.pipeline, boundary.binding, boundary.config)
    assert len(summary) == 8 and all(type(value) is str for value in summary)
    assert summary[:6] == [
        'transformer:weight_format=nvfp4;compute_dtype=bfloat16;load_device=cpu',
        'text_encoder:quantization=bnb_4bit;quant_type=nf4;compute_dtype=bfloat16',
        'text_encoder:double_quant=true;quant_storage=uint8;load_device=cuda:0',
        'text_encoder:skip_modules=model.visual,lm_head',
        'text_encoder:converted_linear_count=252',
        'text_encoder:placement=model_cpu;current_device=cpu',
    ]
    assert summary[6] == 'text_encoder:recipe_id=' + policy.TEXT_ENCODER_NF4_RECIPE_ID
    assert summary[7] == 'text_encoder:signature=' + boundary.binding.text_encoder_quantization['signature']
    record = capture_generation_inputs('models', graph()['models'], {'quant_config': summary})
    assert record['fields']['quant_config'] == {'value': summary, 'source': 'connected',
        'sourceNodeId': 'quant', 'sourcePortId': 'quantization_config'}
    assert record['omittedFields'] == {}
    assert boundary.config['text_encoder'] is original_text


@pytest.mark.parametrize('change', ['missing-text', 'extra-config', 'foreign-bfl', 'bfl-dtype',
    'text-double-quant', 'wrong-weight-format', 'actual-bfl-dtype', 'actual-packed-format', 'owner-binding'])
def test_consumed_summary_refuses_actual_config_or_owner_mismatch(consumed_quantization_boundary, change):
    import torch
    boundary = consumed_quantization_boundary
    if change == 'missing-text':
        boundary.config.pop('text_encoder')
    elif change == 'extra-config':
        boundary.config['vae'] = object()
    elif change == 'foreign-bfl':
        boundary.config['transformer'] = object()
    elif change == 'bfl-dtype':
        boundary.config['transformer'].compute_dtype = torch.float16
    elif change == 'text-double-quant':
        boundary.config['text_encoder'].bnb_4bit_use_double_quant = False
    elif change == 'wrong-weight-format':
        boundary.binding.recipe['weightFormat'] = 'fp8r'
    elif change == 'actual-bfl-dtype':
        boundary.pipeline.transformer.hf_quantizer.compute_dtype = torch.float16
    elif change == 'actual-packed-format':
        boundary.packed[:] = [boundary.fp8()]
    else:
        boundary.pipeline._modiff_weight_format_binding = None
    with pytest.raises(ValueError):
        policy.text_encoder_nf4_consumed_quantization(boundary.pipeline, boundary.binding, boundary.config)
