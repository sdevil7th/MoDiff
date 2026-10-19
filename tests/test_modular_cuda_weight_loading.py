"""Actual native loader functions with CPU stubs: no ML/device imports."""

import ast
from copy import deepcopy
import hashlib
import importlib
import json
import logging
from pathlib import Path
import struct
import sys
import traceback
from types import ModuleType, SimpleNamespace

import pytest

from modiff.artifact_weight_memory import artifact_parameter_conversion_reserve, artifact_parameter_lower_bound


SOURCE = Path(__file__).parents[1] / 'modules/ModularDiffusers/loaders.py'
GIB = 1024**3
REVISION = 'a' * 40
CLASSES = {'transformer': ('diffusers', 'ZImageTransformer2DModel'),
           'vae': ('diffusers', 'AutoencoderKL'), 'text_encoder': ('transformers', 'Qwen3Model')}


class FakeModule:
    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = None

    def parameters(self):
        return iter([SimpleNamespace(dtype='bfloat16', device='cuda:0')])

    def modules(self):
        return iter([self])


@pytest.fixture
def loader(tmp_path, monkeypatch):
    root = tmp_path / 'hub' / 'models--test--contract' / 'snapshots' / REVISION
    root.mkdir(parents=True)
    (root / 'model_index.json').write_text(json.dumps(CLASSES))
    official = {}
    for name, (library, class_name) in CLASSES.items():
        folder = root / name
        folder.mkdir()
        config = {'model_type': 'qwen3'} if library == 'transformers' else {'_class_name': class_name}
        (folder / 'config.json').write_text(json.dumps(config))
        raw = json.dumps({'matrix': {'shape': [4], 'dtype': 'F32', 'data_offsets': [0, 16]}}).encode()
        (folder / 'weights.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(16))
        official[name] = type(class_name, (FakeModule,), {})
    for library in ('diffusers', 'transformers'):
        module = ModuleType(library)
        for name, (selected, class_name) in CLASSES.items():
            if selected == library:
                setattr(module, class_name, official[name])
        monkeypatch.setitem(sys.modules, library, module)

    status = {'complete': True, 'exactRevisionComplete': True,
              'snapshots': [{'complete': True, 'path': str(root)}]}
    inventory = ModuleType('utils.huggingface')
    inventory.get_local_models = lambda: [{'id': 'test/contract', 'cache_dirs': ['other-cache']}]
    monkeypatch.setitem(sys.modules, 'utils.huggingface', inventory)
    auto = ModuleType('modiff.auto_resource')
    auto.artifact_revision_cache_status = lambda *args: deepcopy(status)
    monkeypatch.setitem(sys.modules, 'modiff.auto_resource', auto)
    bound = artifact_parameter_lower_bound(status, REVISION, 'bfloat16')
    reserve = artifact_parameter_conversion_reserve(status, REVISION, 'bfloat16')
    context = {'snapshot': root, 'repository': 'test/contract', 'revision': REVISION,
               'bound': bound, 'reserve': reserve, 'parallel': False, 'configs': {}}
    hardware = {'system': {'ram_total': 32 * GIB, 'ram_available': 24 * GIB},
                'devices': [{'device': 'cuda:0', 'backend': 'cuda', 'memory_kind': 'dedicated',
                             'planning_memory_total': 16 * GIB, 'planning_memory_free': 15 * GIB}]}
    hardware_module = ModuleType('modiff.hardware')
    def snapshot(**kwargs):
        assert kwargs['refresh'] is True
        return deepcopy(hardware)
    hardware_module.get_hardware_snapshot = snapshot
    monkeypatch.setitem(sys.modules, 'modiff.hardware', hardware_module)
    flags = ModuleType('modiff.runtime_environment')
    flags.diffusers_parallel_loading_enabled = lambda: False
    monkeypatch.setitem(sys.modules, 'modiff.runtime_environment', flags)
    events = []
    namespace = {'Path': Path, 'hashlib': hashlib, 'json': json, 'importlib': importlib,
                 'torch': SimpleNamespace(nn=SimpleNamespace(Module=FakeModule),
                                          cuda=SimpleNamespace(is_available=lambda: False)),
                 'MAX_REVIEWED_PIPELINE_INDEX_BYTES': 1024**2, 'MAX_REVIEWED_JSON_ITEMS': 100_000,
                 'MAX_REVIEWED_JSON_DEPTH': 64, 'CUSTOM_PIPELINE_MODEL_TYPE': 'CustomPipeline',
                 'OFFLOAD_MODE_GROUP_DISK': 'group_disk', 'GROUP_OFFLOAD_COMPONENTS': set(CLASSES),
                 'resolve_snapshot_cache_file': lambda path, **kwargs: path,
                 '_containing_hf_cache_root': lambda path: root.parents[2],
                 'exact_cached_snapshot_path': lambda *args: root,
                 'diffusers_logging': SimpleNamespace(tqdm=lambda values, **kwargs: values),
                 'hub_constants': SimpleNamespace(HF_HUB_OFFLINE=False),
                 'logger': logging.getLogger(__name__), 'traceback': traceback,
                 'safe_diagnostic_value': lambda value: value,
                 '_processor_model_config_cache_state': lambda *args, **kwargs: 'present',
                 '_processor_model_config_binding': lambda *args, **kwargs: None,
                 'component_quant_config_summary': lambda value: {}}
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and (node.name.startswith('_cuda_weight_loading') or node.name in {
                 '_prepare_cuda_weight_loading', '_validate_cuda_weight_loading_specs',
                 '_assert_cuda_weight_loading_capacity', 'load_components_strict',
                 'component_load_kwargs_for', 'RequiredComponentLoadError', '_reject_duplicate_pipeline_index_keys',
                 '_reject_nonfinite_pipeline_index_number', '_validate_reviewed_json_shape'})]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)

    class Spec:
        def __init__(self, name):
            self.name = self.subfolder = name
            self.type_hint = official[name]
            self.default_creation_method = 'from_pretrained'
            self.pretrained_model_name_or_path = 'test/contract'
            self.revision = REVISION
            self.variant = self.config = None
        def load(self, **kwargs):
            events.append(('load', self.name, kwargs))
            return self.type_hint()

    specs = {name: Spec(name) for name in CLASSES}
    pipeline = SimpleNamespace(_component_specs=specs, _pretrained_model_name_or_path='test/contract')
    registered = {}
    def register(**kwargs):
        registered.update(kwargs)
        events.append(('register', next(iter(kwargs))))
    pipeline.register_components = register
    def hooks(pipeline, **kwargs):
        name = kwargs['component_names'][0]
        events.append(('hook', name))
        return SimpleNamespace(applied=True, method='group_offload', mode='group_disk', components=[name], disk_path='local')
    namespace['apply_component_group_offload'] = hooks
    def run(*, required_names=None, **kwargs):
        return namespace['load_components_strict'](
            pipeline, list(specs), required_names=set(specs) if required_names is None else required_names,
            model_id='test/contract', dtype='bfloat16',
            offload_mode='group_disk', quant_config=None, diagnostics={}, component_load_kwargs={'torch_dtype': 'bfloat16'},
            incremental_group_offload={'mode': 'group_disk', 'device': 'cuda:0', 'node_id': 'models', 'scope': 'modular-diffusers'},
            cuda_weight_loading=context, **kwargs)
    return SimpleNamespace(ns=namespace, context=context, hardware=hardware, pipeline=pipeline, run=run,
                           events=events, root=root, flags=flags, official=official, status=status, registered=registered)


def test_official_load_kwargs_and_immediate_hooks_precede_the_next_component(loader):
    loader.run()
    assert [(event[0], event[1]) for event in loader.events] == [
        (action, name) for name in CLASSES for action in ('load', 'register', 'hook')]
    for _, _, kwargs in (event for event in loader.events if event[0] == 'load'):
        assert kwargs == {'torch_dtype': 'bfloat16', 'device_map': 'cuda:0', 'low_cpu_mem_usage': True,
                          'local_files_only': True, 'use_safetensors': True,
                          'cache_dir': str(loader.root.parent.parent.parent)}


@pytest.mark.parametrize('field,value', [('offload_mode', 'model_cpu'), ('device', 'cuda:1'), ('device', 'cpu:0'),
    ('dtype', 'float16'), ('model_type', 'CustomPipeline'), ('lora_list', []), ('lora_list', [{}]),
    ('quant_config', {}), ('unet', {}), ('vae', {}), ('controlnet', {}), ('weight_variant', 'fp16')])
def test_unsupported_controls_refuse_before_allocation(loader, field, value):
    kwargs = dict(model_type='ZImageModularPipeline', offload_mode='group_disk', device='cuda:0', dtype='bfloat16',
                  lora_list=None, quant_config=None, unet=None, vae=None, controlnet=None, weight_variant=None)
    kwargs[field] = value
    with pytest.raises(ValueError, match='CUDA weight loading requires'):
        loader.ns['_cuda_weight_loading_controls'](**kwargs)
    assert loader.events == []


@pytest.mark.parametrize('field,value', [('revision', 'b' * 40), ('subfolder', 'other'), ('variant', 'fp16'),
    ('config', {}), ('default_creation_method', 'from_config'), ('pretrained_model_name_or_path', 'other/repo'),
    ('type_hint', FakeModule)])
def test_even_last_component_contract_is_checked_before_any_weight_load(loader, field, value):
    setattr(loader.pipeline._component_specs['text_encoder'], field, value)
    with pytest.raises(ValueError, match='unreviewed class or loading contract'):
        loader.run()
    assert loader.events == []


@pytest.mark.parametrize('field,value', [('quantization_config', {'bits': 4}), ('auto_map', {'AutoModel': 'remote.Model'}),
    ('dtype_plan', {'layers': 'float32'}), ('_keep_in_fp32_modules', ['layers'])])
def test_checkpoint_configuration_cannot_silently_expand_the_uniform_dtype_contract(loader, field, value):
    path = loader.root / 'text_encoder/config.json'
    path.write_text(json.dumps({'model_type': 'qwen3', field: value}))
    with pytest.raises(ValueError, match='does not admit'):
        loader.run()
    assert loader.events == []


def test_forward_upcast_policy_is_not_a_checkpoint_loading_dtype_override(loader):
    (loader.root / 'vae/config.json').write_text(json.dumps({'_class_name': 'AutoencoderKL', 'force_upcast': True}))
    loader.run()


@pytest.mark.parametrize('field', ['_keep_in_fp32_modules', '_keep_in_fp32_modules_strict', '_dtype_plan'])
def test_actual_official_class_loading_dtype_overrides_refuse_before_weights(loader, field):
    setattr(loader.official['vae'], field, ['layers'])
    with pytest.raises(ValueError, match='unreviewed class'):
        loader.run()
    assert loader.events == []


@pytest.mark.parametrize('location,key,value', [('system', 'ram_available', 0), ('system', 'ram_available', None),
    ('system', 'ram_available', True), ('device', 'planning_memory_free', 0), ('device', 'planning_memory_free', None),
    ('device', 'planning_memory_free', True), ('device', 'device', 'cuda:1'), ('device', 'backend', 'rocm'),
    ('device', 'memory_kind', 'shared')])
def test_unknown_zero_and_wrong_scope_capacity_never_become_total_memory_authority(loader, location, key, value):
    target = loader.hardware['system'] if location == 'system' else loader.hardware['devices'][0]
    target[key] = value
    with pytest.raises(ValueError, match='CUDA weight loading'):
        loader.run()
    assert loader.events == []


@pytest.mark.parametrize('pool,required', [('system', 4 * GIB + 24), ('device', 2 * GIB + 8)])
def test_actual_free_capacity_boundary_is_exact_and_headroom_is_additional(loader, pool, required):
    target = loader.hardware['system'] if pool == 'system' else loader.hardware['devices'][0]
    key = 'ram_available' if pool == 'system' else 'planning_memory_free'
    target[key] = required - 1
    with pytest.raises(ValueError, match='capacity refused'):
        loader.run()
    assert loader.events == []
    target[key] = required
    loader.run()


def test_captured_parallel_flag_drift_refuses_without_loading(loader):
    loader.flags.diffusers_parallel_loading_enabled = lambda: True
    with pytest.raises(ValueError, match='captured library loading policy changed'):
        loader.run()
    assert loader.events == []


def test_failed_or_missing_immediate_hook_never_loads_another_component(loader):
    def refused(*args, **kwargs):
        return SimpleNamespace(applied=False)
    loader.ns['apply_component_group_offload'] = refused
    with pytest.raises(RuntimeError, match='immediate official disk hooks'):
        loader.run()
    assert [event[1] for event in loader.events if event[0] == 'load'] == ['transformer']
    assert loader.registered['transformer']._modiff_modular_runtime_policy == {'configuration_failed': True}


def test_cuda_load_failure_never_retries_the_processor_config_fallback(loader):
    def fail(**kwargs):
        loader.events.append(('load', 'transformer', kwargs))
        raise RuntimeError('original official loader failure')
    loader.pipeline._component_specs['transformer'].load = fail
    loader.ns['_processor_model_config_cache_state'] = lambda *args, **kwargs: 'unknown'
    with pytest.raises(RuntimeError, match='original official loader failure'):
        loader.run()
    assert len(loader.events) == 1


def test_cpu_strict_load_keeps_historical_kwargs_and_does_not_run_staging_guards(loader):
    loader.ns['_validate_cuda_weight_loading_specs'] = lambda *args: pytest.fail('CPU entered CUDA guard')
    loader.ns['load_components_strict'](loader.pipeline, ['transformer'], required_names={'transformer'},
        model_id='test/contract', dtype='bfloat16', offload_mode='none', quant_config=None, diagnostics={},
        component_load_kwargs={'torch_dtype': 'bfloat16'})
    assert loader.events[0][2] == {'torch_dtype': 'bfloat16'}


@pytest.mark.parametrize('change', ['none', 'incomplete', 'different_snapshot', 'invalid_header'])
def test_census_uses_real_primary_cache_validation_and_never_another_complete_cache(loader, monkeypatch, change):
    cache_root = str(loader.root.parent.parent.parent)
    inventory = ModuleType('utils.huggingface')
    inventory.get_local_models = lambda: [{'id': 'test/contract', 'cache_dirs': ['other-cache']}]
    monkeypatch.setitem(sys.modules, 'utils.huggingface', inventory)
    auto = ModuleType('modiff.auto_resource')
    status = deepcopy(loader.status)
    if change == 'incomplete':
        status['exactRevisionComplete'] = False
    elif change == 'different_snapshot':
        status['snapshots'][0]['path'] = str(loader.root.parent / ('b' * 40))
    elif change == 'invalid_header':
        (loader.root / 'transformer/weights.safetensors').write_bytes(b'invalid')
    def validated(repository, revision, records):
        assert (repository, revision) == ('test/contract', REVISION)
        assert records[0]['cache_dirs'] == [cache_root]
        assert records[0]['cache_dir'] == cache_root
        return status
    auto.artifact_revision_cache_status = validated
    monkeypatch.setitem(sys.modules, 'modiff.auto_resource', auto)
    if change == 'none':
        assert loader.ns['_cuda_weight_loading_census'](loader.root, 'test/contract', REVISION) == (
            loader.context['bound'], loader.context['reserve'])
    else:
        with pytest.raises(ValueError, match='CUDA weight loading requires'):
            loader.ns['_cuda_weight_loading_census'](loader.root, 'test/contract', REVISION)
    assert loader.events == []


def test_actual_declared_native_image_profile_is_required_not_a_namespace_or_class_name(loader):
    loader.ns['_cuda_weight_loading_census'] = lambda *args: (loader.context['bound'], loader.context['reserve'])
    context = loader.ns['_prepare_cuda_weight_loading'](
        model_type='ZImageModularPipeline', repository='Tongyi-MAI/Z-Image-Turbo', revision=REVISION,
        index_filename='model_index.json')
    assert context['bound'] == loader.context['bound']
    for model_type, repository in [('ZImageModularPipeline', 'unreviewed/model'),
                                   ('MiniMaxMusic2ModularPipeline', 'MiniMaxAI/MiniMax-Music-2'),
                                   ('FakeModularImagePipeline', 'Tongyi-MAI/Z-Image-Turbo')]:
        with pytest.raises(ValueError, match='exact declared built-in native image profile'):
            loader.ns['_prepare_cuda_weight_loading'](model_type=model_type, repository=repository,
                                                      revision=REVISION, index_filename='model_index.json')
    assert loader.events == []


def test_configuration_drift_between_components_stops_before_the_affected_official_load(loader):
    original = loader.ns['apply_component_group_offload']
    def hooks(*args, **kwargs):
        result = original(*args, **kwargs)
        (loader.root / 'vae/config.json').write_text(json.dumps({'_class_name': 'AutoencoderKL', 'changed': True}))
        return result
    loader.ns['apply_component_group_offload'] = hooks
    with pytest.raises(ValueError, match='component config changed'):
        loader.run()
    assert [event[1] for event in loader.events if event[0] == 'load'] == ['transformer']


def test_live_capacity_is_reobserved_after_each_official_hook(loader):
    original = loader.ns['apply_component_group_offload']
    def hooks(*args, **kwargs):
        result = original(*args, **kwargs)
        loader.hardware['devices'][0]['planning_memory_free'] = 0
        return result
    loader.ns['apply_component_group_offload'] = hooks
    with pytest.raises(ValueError, match='capacity refused'):
        loader.run()
    assert [event[1] for event in loader.events if event[0] == 'load'] == ['transformer']


@pytest.mark.parametrize('field,value', [('device', 'cpu:0'), ('dtype', 'float32')])
def test_an_official_loader_returning_unexpected_materialization_cannot_install_hooks(loader, field, value):
    model_class = loader.official['transformer']
    model_class.parameters = lambda self: iter([SimpleNamespace(**{
        'device': value if field == 'device' else 'cuda:0', 'dtype': value if field == 'dtype' else 'bfloat16'})])
    with pytest.raises(RuntimeError, match='did not materialize uniform'):
        loader.run()
    assert [(event[0], event[1]) for event in loader.events] == [('load', 'transformer')]


def test_header_drift_between_components_refuses_before_next_weight_allocation(loader):
    original = loader.ns['apply_component_group_offload']
    def hooks(*args, **kwargs):
        result = original(*args, **kwargs)
        path = loader.root / 'vae/weights.safetensors'
        raw = json.dumps({'matrix': {'shape': [8], 'dtype': 'F32', 'data_offsets': [0, 32]}}).encode()
        path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(32))
        return result
    loader.ns['apply_component_group_offload'] = hooks
    with pytest.raises(ValueError, match='header census changed'):
        loader.run()
    assert [event[1] for event in loader.events if event[0] == 'load'] == ['transformer']


@pytest.mark.parametrize('pool', ['system', 'device'])
def test_free_capacity_above_total_refuses_instead_of_granting_impossible_loading_credit(loader, pool):
    target = loader.hardware['system'] if pool == 'system' else loader.hardware['devices'][0]
    total = 'ram_total' if pool == 'system' else 'planning_memory_total'
    free = 'ram_available' if pool == 'system' else 'planning_memory_free'
    target[free] = target[total] + 1
    with pytest.raises(ValueError, match='exceed their physical totals'):
        loader.run()
    assert loader.events == []


@pytest.mark.parametrize('pool', ['system', 'device'])
def test_large_machine_percentage_headroom_remains_stronger_than_the_fixed_floor(loader, pool):
    target = loader.hardware['system'] if pool == 'system' else loader.hardware['devices'][0]
    total = 'ram_total' if pool == 'system' else 'planning_memory_total'
    free = 'ram_available' if pool == 'system' else 'planning_memory_free'
    target[total] = 100 * GIB
    demand = 24 if pool == 'system' else 8
    target[free] = 10 * GIB + demand - 1
    with pytest.raises(ValueError, match='capacity refused'):
        loader.run()
    assert loader.events == []
    target[free] += 1
    loader.run()


@pytest.mark.parametrize('failure', ['load', 'dtype', 'device'])
def test_every_staged_weight_failure_is_fatal_even_when_not_in_required_names(loader, failure):
    if failure == 'load':
        def fail(**kwargs):
            loader.events.append(('load', 'transformer', kwargs))
            raise RuntimeError('staged optional official loader failure')
        loader.pipeline._component_specs['transformer'].load = fail
        message = 'staged optional official loader failure'
    else:
        loader.official['transformer'].parameters = lambda self: iter([SimpleNamespace(
            dtype='float32' if failure == 'dtype' else 'bfloat16',
            device='cpu:0' if failure == 'device' else 'cuda:0')])
        message = 'did not materialize uniform'
    with pytest.raises(loader.ns['RequiredComponentLoadError'], match=message):
        loader.run(required_names={'vae', 'text_encoder'})
    assert [(event[0], event[1]) for event in loader.events] == [('load', 'transformer')]
    assert loader.registered == {}


def test_cpu_optional_load_failure_keeps_ordinary_diagnostic_continuation(loader):
    diagnostics = {}
    def fail(**kwargs):
        loader.events.append(('load', 'transformer', kwargs))
        raise RuntimeError('ordinary CPU optional failure')
    loader.pipeline._component_specs['transformer'].load = fail
    loader.ns['load_components_strict'](loader.pipeline, ['transformer'], required_names=set(),
        model_id='test/contract', dtype='bfloat16', offload_mode='none', quant_config=None, diagnostics=diagnostics,
        component_load_kwargs={'torch_dtype': 'bfloat16'})
    assert diagnostics['components_failed'][0]['required'] is False
    assert diagnostics['components_failed'][0]['error'] == 'ordinary CPU optional failure'
    assert [(event[0], event[1]) for event in loader.events] == [('load', 'transformer')]
    assert loader.registered == {}
