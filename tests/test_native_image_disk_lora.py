"""Native image disk LoRA uses actual pinned PEFT and disk hooks, no downloads."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import weakref

import pytest
import torch
from diffusers import ComponentsManager, ModelMixin
from diffusers.hooks import apply_group_offloading
from diffusers.loaders import PeftAdapterMixin
from safetensors.torch import load_file, save_file

from modiff.auxiliary_lora import build_lora_descriptor
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES

REPO = 'Tongyi-MAI/Z-Image-Turbo'
REVISION = 'f332072aa78be7aecdf3ee76d5c247082da564a6'


class TinyAdapterModel(ModelMixin, PeftAdapterMixin):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
        with torch.no_grad():
            self.blocks[0].weight.copy_(torch.eye(2))

    def forward(self, value):
        return self.blocks[0](value)

    def get_memory_footprint(self):
        return sum(value.numel() * value.element_size() for value in self.parameters())

    def enable_slicing(self):
        self.slicing = True

    def disable_slicing(self):
        self.slicing = False


class TinySpec:
    default_creation_method = 'from_pretrained'
    pretrained_model_name_or_path = REPO
    revision = REVISION
    type_hint = torch.nn.Module

    def __init__(self, name, events):
        self.name, self.events = name, events
        self.load_id = 'tiny-' + name

    def load(self, **kwargs):
        self.events.append(('load', self.name))
        model = TinyAdapterModel()
        model._diffusers_load_id = self.load_id
        return model


class TinyPipeline:
    _lora_loadable_modules = ['transformer']
    hf_device_map = None
    pretrained_component_names = ['transformer', 'text_encoder', 'vae']

    def __init__(self, manager, owner, events):
        self.manager, self.owner, self.events = manager, owner, events
        self._component_specs = {name: TinySpec(name, events) for name in self.pretrained_component_names}
        self.registered = {}
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler()

    @property
    def components(self):
        return self.registered

    def get_component_spec(self, name):
        return self._component_specs[name]

    def register_components(self, **values):
        for name, value in values.items():
            self.manager.add(name, value, collection=self.owner)
        self.update_components(**values)

    def update_components(self, **values):
        self.registered.update(values)
        for name, value in values.items():
            setattr(self, name, value)

    def get_list_adapters(self):
        return {'transformer': list(getattr(self.transformer, 'peft_config', {}))}

    def load_lora_weights(self, directory, *, weight_name, adapter_name, use_safetensors):
        assert use_safetensors
        self.events.append(('lora', adapter_name))
        self.transformer.load_lora_adapter(load_file(Path(directory) / weight_name), prefix=None,
                                           adapter_name=adapter_name, _pipeline=self)

    def set_adapters(self, names, weights):
        self.transformer.set_adapters(names, weights)

    def unload_lora_weights(self):
        self.transformer.unload_lora()


def descriptor(tmp_path, value=1.0, *, scale=1.0, name='style', scheduler_class='', scheduler_config=None):
    directory = tmp_path / ('adapter-' + str(value))
    directory.mkdir(exist_ok=True)
    file = directory / 'adapter.safetensors'
    save_file({'blocks.0.lora_A.weight': torch.tensor([[value, value]]),
               'blocks.0.lora_B.weight': torch.tensor([[value], [value]])}, file)
    return build_lora_descriptor(selection={'source': 'local', 'value': str(directory)},
        weight_name=file.name, revision=None, expected_sha256=None, adapter_name=name, scale=scale,
        scheduler_class=scheduler_class, scheduler_config=scheduler_config)


@pytest.fixture
def native_runtime(tmp_path, monkeypatch):
    from modiff.config import CONFIG
    for field in ('work_dir', 'data', 'images', 'videos', 'audio', 'models', 'upscalers', 'temp'):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(tmp_path / 'hub'))
    monkeypatch.setitem(CONFIG.hf, 'online_status', 'Offline')
    monkeypatch.delenv('MODIFF_SUPERVISOR_QUEUE_STATE', raising=False)
    from modules.ModularDiffusers import loaders
    import modules.ModularDiffusers as modular

    manager, events, pipelines = ComponentsManager(), [], []
    monkeypatch.setattr(modular, 'components', manager)
    monkeypatch.setattr(loaders, 'components', manager)
    monkeypatch.setattr(torch.accelerator, 'current_accelerator', lambda: torch.device('cpu'))
    monkeypatch.setattr(loaders.ModelsLoader, '_effective_builtin_selector',
                        lambda self, **kwargs: (kwargs['repo_id'], kwargs['revision']))
    monkeypatch.setattr(loaders.ModelsLoader, '_preflight_reviewed_builtin_selection',
                        lambda self, **kwargs: ('hub', REPO, REVISION, 'model_index.json', {}))
    monkeypatch.setattr(loaders, '_primary_component_cache_dirs', lambda *args: {})
    monkeypatch.setattr(loaders, 'pin_modular_component_revisions', lambda *args: {})
    monkeypatch.setattr(loaders, '_loader_text_encoder_component_names', lambda pipeline: ['text_encoder'])
    monkeypatch.setattr(loaders, '_reviewed_loader_component_outputs', lambda model_type: ())
    monkeypatch.setattr(loaders, 'annotate_modular_loader_outputs', lambda *args, **kwargs: None)
    monkeypatch.setattr(loaders, 'bind_loader_outputs', lambda *args, **kwargs: None)

    def instantiate(*args, collection, **kwargs):
        pipeline = TinyPipeline(manager, collection, events)
        pipelines.append(weakref.ref(pipeline))
        return pipeline

    monkeypatch.setattr(loaders, '_instantiate_reviewed_builtin_pipeline', instantiate)

    def offload(pipeline, *, component_names, **kwargs):
        applied = []
        for name in component_names:
            model = getattr(pipeline, name, None)
            if not isinstance(model, torch.nn.Module) or hasattr(model, '_diffusers_hook'):
                continue
            events.append(('hook', name))
            path = tmp_path / f'disk-{len(events)}-{name}'
            path.mkdir()
            apply_group_offloading(model, onload_device='cpu', offload_device='cpu',
                offload_type='block_level', num_blocks_per_group=1, use_stream=False,
                low_cpu_mem_usage=True, offload_to_disk_path=str(path))
            applied.append(name)
        # Existing already installed hooks also count as a valid full pass.
        names = [name for name in component_names if isinstance(getattr(pipeline, name, None), torch.nn.Module)]
        return SimpleNamespace(applied=bool(names), components=names, mode='group_disk',
                               method='group_offload', disk_path=None, detail=None)

    monkeypatch.setattr(loaders, 'apply_component_group_offload', offload)
    return loaders, manager, events, pipelines


def native_args(loras=None):
    return dict(repo_id={'source': 'hub', 'value': REPO}, revision=REVISION,
                model_type='ZImageModularPipeline', dtype=torch.float32, device='cuda:0',
                offload_mode='group_disk', auto_offload=True, lora_list=loras)


def test_actual_native_loader_injects_lora_before_target_disk_files_exist(native_runtime, tmp_path):
    loaders, _, events, _ = native_runtime
    node = loaders.ModelsLoader('models')
    node.execute(**native_args([descriptor(tmp_path)]))
    result = node.loader.transformer(torch.tensor([[1.0, 2.0]])).detach()
    assert result.tolist() == [[4.0, 5.0]]
    assert events.index(('lora', 'style')) < events.index(('hook', 'transformer'))
    assert events.index(('hook', 'text_encoder')) < events.index(('load', 'transformer'))


def test_reviewed_native_image_lora_targets_equal_pinned_official_classes():
    import diffusers
    from modiff.native_image_auto_placement import declared_image_owner

    for profile in DIFFUSERS_EXECUTION_PROFILES.values():
        if (profile.execution_path != 'modular-diffusers'
                or not any(declared_image_owner(profile, profile.default_repo, mode) for mode in profile.modes)):
            continue
        pipeline_class = getattr(diffusers, profile.pipeline_class, None)
        if pipeline_class is None:
            assert profile.pipeline_class in {'Flux3ModularPipeline', 'Flux3DistilledModularPipeline'}
            legacy_ids = {'flux3-klein:modular', 'flux3-klein-base:modular'}
            public_ids = {'flux3-klein:official-provider:v1', 'flux3-klein-base:official-provider:v1'}
            assert profile.id in legacy_ids | public_ids
            assert profile.public is (profile.id in public_ids)
            assert profile.live_proof is False
            if profile.id in legacy_ids:
                assert profile.required_diffusers_provider is None
            else:
                assert profile.required_diffusers_provider == 'huggingface-diffusers-flux3-7ecb1e1'
                assert profile.replaces_private_authoring_profile == profile.id.replace(':official-provider:v1', ':modular')
                requirement = profile.to_public_dict()['optionalRuntimeRequirement']
                assert requirement['requiredNow'] is True and requirement['state'] != 'active'
            assert profile.component_lora_declaration is not None
            assert profile.component_lora_declaration.profile_id == profile.id
            assert profile.lora_loadable_components == ('transformer',)
            continue  # A declaration does not supply the absent official class or qualify execution.
        actual = tuple(getattr(pipeline_class, '_lora_loadable_modules', ()))
        assert profile.lora_loadable_components == actual, profile.id
        assert 'lora_loadable_components' not in profile.to_public_dict()


@pytest.fixture
def graph_runtime(native_runtime, tmp_path, monkeypatch):
    import modules
    import modules.Text.main as text
    from modiff.NodeBase import NodeBase
    from modiff.server import WebServer, memory_manager

    loaders, manager, events, pipelines = native_runtime
    monkeypatch.setattr(memory_manager, 'cache', {})
    monkeypatch.setattr('modiff.NodeBase.modelstore.is_hf_cached', lambda *args: True)
    monkeypatch.setattr('modiff.NodeBase.modelstore.is_local_cached', lambda *args: True)
    monkeypatch.setattr('modiff.server.get_hardware_snapshot', lambda *args, **kwargs: {
        'system': {'ram_total': 64 * 1024**3, 'ram_available': 48 * 1024**3}, 'devices': [],})

    class String(NodeBase):
        def execute(self, bundle, prompt):
            model = manager.get_one(component_id=bundle['transformer']['model_id'])
            return {'value': model(torch.tensor([[1.0, 2.0]])).detach().tolist()}

    String.__module__ = 'modules.Text.main'
    monkeypatch.setattr(text, 'String', String, raising=False)
    monkeypatch.setitem(modules.MODULE_MAP['modules.Text'], 'String', {
        'params': {'bundle': {'type': 'diffusers_modular_pipeline_components'}, 'prompt': {'type': 'string'}},
        'skipParamsCheck': True, 'outputs': ['value'],
    })
    server = WebServer(modules=modules.MODULE_MAP, work_dir=str(tmp_path), data_dir=str(tmp_path / 'data'))
    monkeypatch.setattr(server, 'queue_message', lambda _: None)
    monkeypatch.setattr(server, '_runtime_fingerprint', lambda **kwargs: {'fingerprint': 'tiny-cpu-only'})
    monkeypatch.setattr(server, '_record_auto_resource_success', lambda *args, **kwargs: None)
    monkeypatch.setattr(server, '_record_optimization_observations', lambda *args, **kwargs: [])
    yield server, loaders, manager, events, pipelines
    server.node_cache.clear()
    manager.components.clear()
    manager.collections.clear()


def executable_graph(loras=None, *, prompt='original'):
    args = native_args(loras)
    args['dtype'] = 'float32'
    return {'sid': 'unit', 'nodes': {
        'models': {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader',
                   'params': {key: {'value': value} for key, value in args.items()}},
        'generate': {'module': 'modules.Text', 'action': 'String', 'params': {
            'bundle': {'sourceId': 'models', 'sourceKey': 'pipeline_components'}, 'prompt': {'value': prompt}}},
    }, 'paths': [['models', 'generate']]}


def execute_graph(server, graph):
    server.current_task = {'task_id': 'tiny-task', 'sid': 'unit', 'status': 'running', 'progress': 0}
    server._execute_graph(deepcopy(graph))
    return server.node_cache['generate'].output['value']


def test_full_graph_warm_prompt_and_scale_replacement_restore(graph_runtime, tmp_path):
    server, _, _, _, pipelines = graph_runtime
    graph = executable_graph([descriptor(tmp_path)])
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    initial = weakref.ref(server.node_cache['models'].loader)
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    graph['nodes']['generate']['params']['prompt']['value'] = 'prompt only'
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    assert server.node_cache['models'].loader is initial() and len(pipelines) == 1
    graph['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, scale=0.5)]
    assert execute_graph(server, graph) == [[2.5, 3.5]]
    assert initial() is None
    graph['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, 2.0)]
    assert execute_graph(server, graph) == [[13.0, 14.0]]
    graph['nodes']['models']['params']['lora_list']['value'] = None
    assert execute_graph(server, graph) == [[1.0, 2.0]]
    graph['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path)]
    assert execute_graph(server, graph) == [[4.0, 5.0]]


@pytest.mark.parametrize('change', ['explicit_invalidation', 'unrelated_loader_input', 'implementation', 'manager_eviction'])
def test_same_lora_policy_loader_callback_reconstructs_before_allocation(graph_runtime, tmp_path, monkeypatch, change):
    server, loaders, manager, _, _ = graph_runtime
    graph = executable_graph([descriptor(tmp_path)])
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    previous = weakref.ref(server.node_cache['models'].loader)
    if change == 'explicit_invalidation':
        server.node_cache['models'].invalidate_cache()
    elif change == 'unrelated_loader_input':
        graph['nodes']['models']['params']['vae_slicing'] = {'value': True}
    elif change == 'implementation':
        original = loaders.ModelsLoader.execute
        def replacement(self, **kwargs):
            return original(self, **kwargs)
        monkeypatch.setattr(loaders.ModelsLoader, 'execute', replacement)
    else:
        component_id = server.node_cache['models'].output['pipeline_components']['transformer']['model_id']
        manager.components.pop(component_id)
    original_factory = loaders._instantiate_reviewed_builtin_pipeline
    def require_destroyed(*args, **kwargs):
        assert previous() is None, 'Old adapted owner survives replacement allocation'
        return original_factory(*args, **kwargs)
    monkeypatch.setattr(loaders, '_instantiate_reviewed_builtin_pipeline', require_destroyed)
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    assert previous() is None


@pytest.mark.parametrize('empty', [None, []])
def test_full_graph_vanilla_add_and_remove_returns_fresh_base(graph_runtime, tmp_path, empty):
    server, _, _, _, pipelines = graph_runtime
    graph = executable_graph(empty)
    assert execute_graph(server, graph) == [[1.0, 2.0]]
    assert execute_graph(server, graph) == [[1.0, 2.0]] and len(pipelines) == 1
    graph['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path)]
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    graph['nodes']['models']['params']['lora_list']['value'] = empty
    assert execute_graph(server, graph) == [[1.0, 2.0]]


def test_connected_loading_control_change_reconstructs_without_early_supplier_execution(graph_runtime, tmp_path, monkeypatch):
    import modules
    import modules.Text.main as text
    from modiff.NodeBase import NodeBase
    server, _, _, _, _ = graph_runtime
    calls = []
    class Control(NodeBase):
        def execute(self, value):
            calls.append(value)
            return {'value': value}
    Control.__module__ = 'modules.Text.main'
    monkeypatch.setattr(text, 'Control', Control, raising=False)
    monkeypatch.setitem(modules.MODULE_MAP['modules.Text'], 'Control',
        {'params': {'value': {'type': 'boolean'}}, 'outputs': ['value'], 'skipParamsCheck': True})
    graph = executable_graph([descriptor(tmp_path)])
    graph['nodes']['control'] = {'module': 'modules.Text', 'action': 'Control', 'params': {'value': {'value': False}}}
    graph['nodes']['models']['params']['vae_slicing'] = {'sourceId': 'control', 'sourceKey': 'value'}
    graph['paths'] = [['control', 'models', 'generate']]
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    previous = weakref.ref(server.node_cache['models'].loader)
    assert execute_graph(server, graph) == [[4.0, 5.0]] and calls == [False]
    graph['nodes']['control']['params']['value']['value'] = True
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    assert previous() is None and calls == [False, True]


def test_failed_official_adapter_injection_never_blesses_next_reuse(graph_runtime, tmp_path, monkeypatch):
    server, _, _, _, _ = graph_runtime
    original = TinyPipeline.load_lora_weights
    def fail_after_mutation(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError('controlled injected failure')
    monkeypatch.setattr(TinyPipeline, 'load_lora_weights', fail_after_mutation)
    graph = executable_graph([descriptor(tmp_path)])
    with pytest.raises(RuntimeError, match='controlled injected failure'):
        execute_graph(server, graph)
    assert server.node_cache['models']._native_disk_lora_state is None
    previous = weakref.ref(server.node_cache['models'].loader)
    monkeypatch.setattr(TinyPipeline, 'load_lora_weights', original)
    assert execute_graph(server, graph) == [[4.0, 5.0]] and previous() is None


def test_shared_target_is_refused_before_mutating_another_owner(graph_runtime, tmp_path):
    server, _, manager, events, _ = graph_runtime
    graph = executable_graph([descriptor(tmp_path)])
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    target = server.node_cache['models'].loader.transformer
    ids = {key for key, value in manager.components.items() if value is target}
    manager.collections['other-owner'] = set(ids)
    count = len([event for event in events if event[0] == 'lora'])
    graph['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, scale=.5)]
    with pytest.raises(RuntimeError, match='exclusively owned|shared'):
        execute_graph(server, graph)
    assert len([event for event in events if event[0] == 'lora']) == count
    assert ids.issubset(manager.collections['other-owner'])


@pytest.mark.parametrize('change', ['name', 'order', 'scheduler'])
def test_full_graph_exact_order_name_and_scheduler_policy_replacement(graph_runtime, tmp_path, change):
    server, _, _, _, _ = graph_runtime
    initial = [descriptor(tmp_path, name='one'), descriptor(tmp_path, 2.0, name='two')]
    graph = executable_graph(initial)
    execute_graph(server, graph)
    previous = weakref.ref(server.node_cache['models'].loader)
    if change == 'name':
        changed = [descriptor(tmp_path, name='renamed'), initial[1]]
    elif change == 'order':
        changed = list(reversed(initial))
    else:
        changed = [descriptor(tmp_path, name='one', scheduler_class='FlowMatchEulerDiscreteScheduler',
                              scheduler_config={'shift': 2.0}), initial[1]]
    graph['nodes']['models']['params']['lora_list']['value'] = changed
    execute_graph(server, graph)
    assert previous() is None
    if change == 'scheduler':
        assert server.node_cache['models'].loader.scheduler.config.shift == 2.0


@pytest.mark.parametrize('targets', [(['transformer'],), ({'name': 'transformer'},), ('transformer','transformer'), ('bad/name',)])
def test_private_reviewed_target_validation_has_consistent_error(targets):
    from dataclasses import replace
    with pytest.raises(ValueError, match='reviewed LoRA'):
        replace(DIFFUSERS_EXECUTION_PROFILES['z-image:modular'], lora_loadable_components=targets)


def test_disk_staging_parameter_floor_sums_all_reviewed_targets_without_peak_claim():
    from modiff.artifact_weight_memory import native_disk_lora_loading_floor
    bound = {'revision': REVISION, 'dtype': 'bfloat16', 'proofScope': 'parameter_storage_lower_bound_only',
             'components': {'transformer': 12, 'text_encoder': 8, 'vae': 2}, 'totalBytes': 22, 'largestComponentBytes': 12}
    assert native_disk_lora_loading_floor(bound, targets=('transformer','text_encoder'), dtype='bfloat16') == {'systemRamBytes': 20, 'vramBytes': 0}
    assert native_disk_lora_loading_floor(bound, targets=('vae',), dtype='bfloat16') == {'systemRamBytes': 12, 'vramBytes': 0}
    for targets in ((), ('missing',), ([],), ('transformer','transformer')):
        assert native_disk_lora_loading_floor(bound, targets=targets, dtype='bfloat16') == {}
    assert native_disk_lora_loading_floor(bound, targets=('transformer',), dtype='float32') == {}


def connected_lora_graph(tmp_path):
    import hashlib
    own = descriptor(tmp_path)
    file = Path(own['artifact']['root']) / own['artifact']['weight_name']
    graph = executable_graph()
    graph['nodes']['style'] = {'module': 'modules.ModularDiffusers', 'action': 'Lora', 'params': {
        key: {'value': value} for key, value in {
            'model': {'source': 'local', 'value': str(file)}, 'weight_name': file.name,
            'expected_sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'adapter_name': 'style', 'scale': 1.0,
        }.items()}}
    graph['nodes']['models']['params']['lora_list'] = {'sourceId': 'style', 'sourceKey': 'lora'}
    graph['paths'] = [['style', 'models', 'generate']]
    return graph


def test_auto_preparation_resets_changed_connected_lora_before_cache_credit_and_replan(graph_runtime, tmp_path):
    from modiff.native_disk_lora import graph_disk_lora_policy
    server, _, _, _, _ = graph_runtime
    graph = connected_lora_graph(tmp_path)
    assert execute_graph(server, graph) == [[4.0, 5.0]]
    previous = weakref.ref(server.node_cache['models'].loader)
    settings = native_args()
    plan = {'loaders': [{'nodeId': 'models', 'settings': settings,
                         'modelType': settings['model_type'], 'repository': REPO}]}
    # Inspection compares literal declared contracts and existing outputs only.
    old = graph_disk_lora_policy(settings, graph, 'models')
    assert old == server.node_cache['models']._native_disk_lora_state['policy']
    # The supplier's flag is still from its FIRST dispatch, not this new run.
    assert server.node_cache['style']._has_changed is True
    assert server._prepare_workflow_auto_owner_policies(graph, plan) is None
    assert execute_graph(server, graph) == [[4.0, 5.0]] and server.node_cache['models'].loader is previous()
    graph['nodes']['style']['params']['scale']['value'] = .5
    released = server._prepare_workflow_auto_owner_policies(graph, plan)
    assert 'models' in released['releasedNodeIds'] and previous() is None
    assert 'models' not in server.node_cache
    assert execute_graph(server, graph) == [[2.5, 3.5]]


def test_runtime_declared_target_mismatch_refuses_before_component_or_lora_loading(native_runtime, tmp_path, monkeypatch):
    loaders, _, events, _ = native_runtime
    monkeypatch.setattr(TinyPipeline, '_lora_loadable_modules', ['text_encoder'])
    with pytest.raises(ValueError, match='reviewed installed pipeline'):
        loaders.ModelsLoader('models').execute(**native_args([descriptor(tmp_path)]))
    assert events == []


def test_disk_target_inspection_is_lazy_and_other_modalities_have_no_policy():
    import subprocess
    import sys
    code = """
import sys
from modiff.native_disk_lora import declared_lora_components, requested_disk_lora_policy
assert declared_lora_components('ZImageModularPipeline', 'Tongyi-MAI/Z-Image-Turbo') == ('transformer',)
assert requested_disk_lora_policy({'model_type': 'AceStepPipeline', 'repo_id': 'ACE-Step/ACE-Step-v1-3.5B', 'device': 'cuda:0', 'offload_mode': 'group_disk'}) is None
assert not {'torch','transformers','peft','diffusers'}.intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_artifact_change_during_actual_loading_leaves_partial_owner_poisoned(graph_runtime, tmp_path, monkeypatch):
    server, _, _, _, _ = graph_runtime
    original = TinyPipeline.load_lora_weights
    def change_after_loading(self, directory, **kwargs):
        original(self, directory, **kwargs)
        save_file({'blocks.0.lora_A.weight': torch.tensor([[3., 3.]]),
                   'blocks.0.lora_B.weight': torch.tensor([[3.], [3.]])}, Path(directory) / kwargs['weight_name'])
    monkeypatch.setattr(TinyPipeline, 'load_lora_weights', change_after_loading)
    graph = executable_graph([descriptor(tmp_path)])
    with pytest.raises(RuntimeError, match='SHA-256|hash|bytes'):
        execute_graph(server, graph)
    assert server.node_cache['models']._native_disk_lora_state is None
    monkeypatch.setattr(TinyPipeline, 'load_lora_weights', original)
    graph['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path)]
    assert execute_graph(server, graph) == [[4., 5.]]


def test_changed_disk_owner_preserves_unrelated_cache_and_material_output(graph_runtime, tmp_path):
    from PIL import Image
    server, _, _, _, _ = graph_runtime
    graph = executable_graph([descriptor(tmp_path)])
    execute_graph(server, graph)
    image = Image.new('RGB', (2, 2), (17, 31, 53))
    peer = SimpleNamespace(output={'images': [image]}, _mm_models=[], _cache_input_sources=[])
    retained = SimpleNamespace(output={'images': [image]}, _mm_models=[], _cache_input_sources=['models'])
    server.node_cache.update(peer=peer, retained=retained)
    changed = native_args([descriptor(tmp_path, scale=.5)])
    result = server._prepare_native_owner_runtime_policies({'models': changed}, 'unit', resolved=True)
    assert 'models' in result['releasedNodeIds']
    assert server.node_cache['peer'] is peer
    assert server.node_cache['retained'].output['images'][0].getpixel((0, 0)) == (17, 31, 53)
    assert not getattr(server.node_cache['retained'], '_cache_valid', False)


def test_conflicting_scheduler_descriptors_refuse_before_owner_release_or_allocation(graph_runtime, tmp_path):
    server, _, _, events, pipelines = graph_runtime
    graph = executable_graph([descriptor(tmp_path)])
    execute_graph(server, graph)
    initial = weakref.ref(server.node_cache['models'].loader)
    changed = [descriptor(tmp_path, name='one', scheduler_class='FlowMatchEulerDiscreteScheduler', scheduler_config={'shift': 2.}),
               descriptor(tmp_path, 2., name='two', scheduler_class='FlowMatchEulerDiscreteScheduler', scheduler_config={'shift': 3.})]
    graph['nodes']['models']['params']['lora_list']['value'] = changed
    before = len(events)
    with pytest.raises(ValueError, match='agree|conflict|scheduler'):
        execute_graph(server, graph)
    assert server.node_cache['models'].loader is initial() and len(pipelines) == 1
    assert len(events) == before


def test_static_native_chain_checks_scheduler_consensus_without_supplier_execution(tmp_path):
    from modiff.native_disk_lora import graph_disk_lora_policy
    graph = connected_lora_graph(tmp_path)
    graph['nodes']['style']['params'].update(scheduler_class={'value': 'FlowMatchEulerDiscreteScheduler'},
                                            scheduler_config={'value': '{"shift":2}'})
    graph['nodes']['style2'] = deepcopy(graph['nodes']['style'])
    graph['nodes']['style2']['params'].update(adapter_name={'value': 'other'}, scheduler_config={'value': '{"shift":3}'},
                                             previous_loras={'sourceId': 'style', 'sourceKey': 'lora'})
    graph['nodes']['models']['params']['lora_list']['sourceId'] = 'style2'
    graph['paths'] = [['style', 'style2', 'models', 'generate']]
    with pytest.raises(ValueError, match='scheduler.*agree|scheduler.*incompatible'):
        graph_disk_lora_policy(native_args(), graph, 'models')


def test_unchanged_connected_lora_auto_graph_retains_owner_through_actual_preamble(graph_runtime, tmp_path, monkeypatch):
    """Real full Auto graph dispatch; only memory recipe admission is synthetic."""
    from modiff.workflow_auto_resource import workflow_graph_hash, workflow_owner_cache_key
    server, _, _, _, pipelines = graph_runtime
    graph = connected_lora_graph(tmp_path)

    def plan(current, **kwargs):
        settings = {**native_args(), 'dtype': 'float32', 'offloadMode': 'group_disk', 'autoOffload': True,
                    'modelType': 'ZImageModularPipeline', 'modelRepo': REPO, 'quantizationMode': 'none'}
        requirements = {'systemRamBytes': 0, 'vramBytes': 0, 'diskFreeBytes': 0}
        return {'schemaVersion': 1, 'graphHash': workflow_graph_hash(current), 'canAutoRun': True,
            'issues': [], 'patches': [], 'adapters': [], 'requirements': requirements,
            'retainedRequirements': requirements, 'schedule': None, 'strategy': 'dependency_order_retained_owners',
            'requiresPreparation': False, 'requiresCachePreparation': False, 'reusedOwnerIds': [],
            'loaders': [{'nodeId': 'models', 'modelType': 'ZImageModularPipeline', 'repository': REPO,
                'candidateId': 'tiny-test-memory-only', 'mode': 'modular_text_to_image', 'settings': settings,
                'cacheKey': workflow_owner_cache_key(current, 'models'), 'requirements': requirements,
                'capacityRequirements': {}, 'workingMemoryPolicy': 'runtime_headroom_policy'}]}

    monkeypatch.setattr(server, '_build_workflow_auto_plan', plan)
    graph['runtimeHints'] = {'resourceMode': 'auto', 'workflowAutoPlan': {'schemaVersion': 1, 'graphHash': workflow_graph_hash(graph)}}
    assert execute_graph(server, graph) == [[4., 5.]]
    initial = weakref.ref(server.node_cache['models'].loader)
    assert server.node_cache['style']._has_changed is True
    assert execute_graph(server, graph) == [[4., 5.]]
    assert server.node_cache['models'].loader is initial() and len(pipelines) == 1


@pytest.mark.parametrize('profile_id', [
    'flux-schnell:modular',
    'flux-krea:modular',
    'sdxl-turbo:modular',
    'flux2-klein-kv:t2i-modular',
])
def test_actual_native_canonical_alias_resolves_own_reviewed_disk_targets(profile_id):
    from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
    from modiff.native_disk_lora import declared_lora_components

    profile = DIFFUSERS_EXECUTION_PROFILES[profile_id]
    selected, reason = resolve_execution_profiles_for_loader(
        'modules.ModularDiffusers', 'ModelsLoader',
        {'model_type': profile.pipeline_class,
         'repo_id': {'source': 'hub', 'value': profile.default_repo}},
    )
    assert reason is None
    assert tuple(item.id for item in selected) == (profile_id,)
    assert declared_lora_components(profile.pipeline_class, profile.default_repo) == profile.lora_loadable_components


@pytest.mark.parametrize('profile_id', ['flux-schnell:modular', 'flux-krea:modular'])
@pytest.mark.parametrize('empty', [None, []])
def test_actual_canonical_two_target_alias_keeps_vanilla_policy_identity(profile_id, empty):
    from modiff.native_disk_lora import requested_disk_lora_policy

    profile = DIFFUSERS_EXECUTION_PROFILES[profile_id]
    settings = {'model_type': profile.pipeline_class,
                'repo_id': {'source': 'hub', 'value': profile.default_repo},
                'device': 'cuda:0', 'auto_offload': True,
                'offload_mode': 'group_disk', 'lora_list': empty}
    assert requested_disk_lora_policy(settings) == (('transformer', 'text_encoder'), ())


@pytest.mark.parametrize('model_type, repository', [
    (None, 'Tongyi-MAI/Z-Image-Turbo'),
    ([], 'Tongyi-MAI/Z-Image-Turbo'),
    ({'model_type': 'ZImageModularPipeline'}, 'Tongyi-MAI/Z-Image-Turbo'),
    (' ZImageModularPipeline', 'Tongyi-MAI/Z-Image-Turbo'),
    ('ZImageModularPipeline', None),
    ('ZImageModularPipeline', []),
    ('ZImageModularPipeline', {'source': 'hub', 'value': 'Tongyi-MAI/Z-Image-Turbo'}),
    ('ZImageModularPipeline', 'Tongyi-MAI/Z-Image-Turbo '),
])
def test_target_metadata_malformed_values_never_become_a_reviewed_selection(model_type, repository):
    from modiff.native_disk_lora import declared_lora_components

    assert declared_lora_components(model_type, repository) is None


def test_single_profile_resolver_shortcut_does_not_grant_foreign_repository_targets():
    from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
    from modiff.native_disk_lora import declared_lora_components

    selected, reason = resolve_execution_profiles_for_loader(
        'modules.ModularDiffusers', 'ModelsLoader',
        {'model_type': 'ZImageModularPipeline',
         'repo_id': {'source': 'hub', 'value': 'foreign/not-reviewed'}},
    )
    assert reason is None and tuple(item.id for item in selected) == ('z-image:modular',)
    assert declared_lora_components('ZImageModularPipeline', 'foreign/not-reviewed') is None


@pytest.mark.parametrize('profile_id', ['flux-schnell:modular', 'flux-krea:modular',
                                        'sdxl-turbo:modular', 'flux2-klein-kv:t2i-modular'])
def test_legacy_logical_alias_target_identity_is_preserved(profile_id):
    from modiff.native_disk_lora import declared_lora_components

    profile = DIFFUSERS_EXECUTION_PROFILES[profile_id]
    assert profile.model_type != profile.pipeline_class
    assert declared_lora_components(profile.model_type, profile.default_repo) == profile.lora_loadable_components


@pytest.mark.parametrize('model_type, repository, targets', [
    ('ZImageModularPipeline', 'Tongyi-MAI/Z-Image-Turbo', ('transformer',)),
    ('StableDiffusionXLModularPipeline', 'stabilityai/stable-diffusion-xl-base-1.0',
     ('unet', 'text_encoder', 'text_encoder_2')),
    ('FluxModularPipeline', 'black-forest-labs/FLUX.1-dev', ('transformer', 'text_encoder')),
    ('FluxModularPipeline', 'black-forest-labs/FLUX.1-dev-FP8', ('transformer', 'text_encoder')),
    ('Cosmos3OmniModularPipeline', 'nvidia/Cosmos3-Nano', ()),
    ('Cosmos3OmniModularPipeline', 'nvidia/Cosmos3-Super-Text2Image', ()),
    ('Cosmos3DistilledModularPipeline', 'nvidia/Cosmos3-Super-Text2Image-4Step', ()),
])
def test_existing_native_target_and_empty_schema_results_are_preserved(model_type, repository, targets):
    from modiff.native_disk_lora import declared_lora_components

    assert declared_lora_components(model_type, repository) == targets


ALIAS_REVISIONS = {
    'flux-schnell:modular': '741f7c3ce8b383c54771c7003378a50191e9efe9',
    'flux-krea:modular': '8162a9c7b05a641be098422bf2fcf335615c2f28',
}


class TinyTwoTargetPipeline(TinyPipeline):
    """Two real PEFT targets under the existing tiny official-hook fixture."""
    _lora_loadable_modules = ['transformer', 'text_encoder']

    def get_list_adapters(self):
        return {name: list(getattr(getattr(self, name), 'peft_config', {}))
                for name in self._lora_loadable_modules}

    def load_lora_weights(self, directory, *, weight_name, adapter_name, use_safetensors):
        assert use_safetensors
        self.events.append(('lora', adapter_name))
        for name in self._lora_loadable_modules:
            getattr(self, name).load_lora_adapter(load_file(Path(directory) / weight_name), prefix=None,
                                                adapter_name=adapter_name, _pipeline=self)

    def set_adapters(self, names, weights):
        for name in self._lora_loadable_modules:
            getattr(self, name).set_adapters(names, weights)

    def unload_lora_weights(self):
        for name in self._lora_loadable_modules:
            getattr(self, name).unload_lora()


def configure_alias_runtime(runtime, monkeypatch, profile_id):
    """Only tiny construction/cache I/O is stubbed; policy and ownership stay real."""
    _, loaders, manager, events, pipelines = runtime
    profile = DIFFUSERS_EXECUTION_PROFILES[profile_id]
    repository, revision = profile.default_repo, ALIAS_REVISIONS[profile_id]
    monkeypatch.setattr(loaders.ModelsLoader, '_preflight_reviewed_builtin_selection',
                        lambda self, **kwargs: ('hub', repository, revision, 'model_index.json', {}))

    def instantiate(*args, collection, **kwargs):
        pipeline = TinyTwoTargetPipeline(manager, collection, events)
        for spec in pipeline._component_specs.values():
            spec.pretrained_model_name_or_path = repository
            spec.revision = revision
        pipelines.append(weakref.ref(pipeline))
        return pipeline

    monkeypatch.setattr(loaders, '_instantiate_reviewed_builtin_pipeline', instantiate)
    return profile, revision


def alias_graph(profile, revision, loras=None):
    graph = executable_graph(loras)
    graph['nodes']['models']['params'].update(
        model_type={'value': profile.pipeline_class},
        repo_id={'value': {'source': 'hub', 'value': profile.default_repo}},
        revision={'value': revision},
    )
    return graph


@pytest.mark.parametrize('profile_id', tuple(ALIAS_REVISIONS))
def test_canonical_flux_alias_injects_both_targets_before_official_disk_hooks(graph_runtime, tmp_path, monkeypatch, profile_id):
    server, _, _, events, _ = graph_runtime
    profile, revision = configure_alias_runtime(graph_runtime, monkeypatch, profile_id)
    graph = alias_graph(profile, revision, [descriptor(tmp_path)])
    assert execute_graph(server, graph) == [[4., 5.]]
    node = server.node_cache['models']
    assert node.loader.text_encoder(torch.tensor([[1., 2.]])).detach().tolist() == [[4., 5.]]
    for name in ('transformer', 'text_encoder'):
        assert events.index(('lora', 'style')) < events.index(('hook', name))
    assert node._native_disk_lora_state['policy'][0] == ('transformer', 'text_encoder')


def target_test_table(monkeypatch, *profiles):
    """Isolate synthetic declaration conflicts; never mutate the real mapping."""
    import modiff.diffusers_profiles as definitions
    import modiff.native_disk_lora as native

    table = {profile.id: profile for profile in profiles}
    monkeypatch.setattr(definitions, 'DIFFUSERS_EXECUTION_PROFILES', table)
    monkeypatch.setattr(native, 'DIFFUSERS_EXECUTION_PROFILES', table)
    return native


@pytest.mark.parametrize('conflicting', [False, True])
def test_canonical_ambiguous_targets_refuse_even_when_declared_tuples_agree(monkeypatch, conflicting):
    from dataclasses import replace

    profile = DIFFUSERS_EXECUTION_PROFILES['flux-schnell:modular']
    peer = replace(profile, id='ambiguous-peer',
                   lora_loadable_components=('transformer',) if conflicting else profile.lora_loadable_components)
    native = target_test_table(monkeypatch, profile, peer)
    assert native.declared_lora_components(profile.pipeline_class, profile.default_repo) == ()
    with pytest.raises(ValueError, match='reviewed immutable component loading targets'):
        native.requested_disk_lora_policy({'model_type': profile.pipeline_class, 'repo_id': profile.default_repo,
                                          'offload_mode': 'group_disk', 'device': 'cuda:0', 'lora_list': [{}]})


def test_canonical_empty_target_schema_refuses_populated_lora_before_descriptor_resolution(monkeypatch):
    from dataclasses import replace

    profile = replace(DIFFUSERS_EXECUTION_PROFILES['flux-schnell:modular'], lora_loadable_components=())
    native = target_test_table(monkeypatch, profile)
    assert native.declared_lora_components(profile.pipeline_class, profile.default_repo) == ()
    settings = {'model_type': profile.pipeline_class, 'repo_id': profile.default_repo,
                'offload_mode': 'group_disk', 'device': 'cuda:0'}
    assert native.requested_disk_lora_policy({**settings, 'lora_list': None}) is None
    with pytest.raises(ValueError, match='reviewed immutable component loading targets'):
        native.requested_disk_lora_policy({**settings, 'lora_list': [{}]})


def test_canonical_foreign_repository_refusal_cannot_fall_back_to_matching_logical_alias(monkeypatch):
    from dataclasses import replace

    alias = DIFFUSERS_EXECUTION_PROFILES['flux-schnell:modular']
    collision = replace(alias, id='canonical-collision', pipeline_class=alias.model_type,
                        default_repo='foreign/reviewed', fallback_repo=None, compatible_repos=(),
                        adapter_authoring=None)
    native = target_test_table(monkeypatch, alias, collision)
    assert native.declared_lora_components(alias.model_type, alias.default_repo) is None


def test_logical_alias_fallback_conflicting_targets_refuse(monkeypatch):
    from dataclasses import replace

    alias = DIFFUSERS_EXECUTION_PROFILES['flux-schnell:modular']
    peer = replace(alias, id='logical-peer', lora_loadable_components=('transformer',))
    native = target_test_table(monkeypatch, alias, peer)
    assert native.declared_lora_components(alias.model_type, alias.default_repo) == ()


def test_foreign_backend_cannot_publish_native_target_schema(monkeypatch):
    alias = DIFFUSERS_EXECUTION_PROFILES['flux-schnell:modular']
    foreign = SimpleNamespace(**vars(alias), backend_path='modules.Other.ModelsLoader')
    native = target_test_table(monkeypatch, foreign)
    assert native.declared_lora_components(alias.model_type, alias.default_repo) is None
    assert native.declared_lora_components(alias.pipeline_class, alias.default_repo) is None


@pytest.mark.parametrize('pipeline_class', ['WanModularPipeline', 'MiniMaxMusic3ModularPipeline'])
def test_nonimage_native_profile_never_grants_image_disk_targets(pipeline_class):
    from modiff.native_disk_lora import declared_lora_components
    from modiff.native_image_auto_placement import declared_image_owner

    profile = next(item for item in DIFFUSERS_EXECUTION_PROFILES.values()
                   if item.pipeline_class == pipeline_class and item.execution_path == 'modular-diffusers')
    assert not any(declared_image_owner(profile, profile.default_repo, mode) for mode in profile.modes)
    assert declared_lora_components(profile.pipeline_class, profile.default_repo) is None


def assert_alias_outputs(server, expected):
    for name in ('transformer', 'text_encoder'):
        actual = getattr(server.node_cache['models'].loader, name)(torch.tensor([[1., 2.]]))
        assert actual.detach().tolist() == expected


@pytest.mark.parametrize('profile_id', tuple(ALIAS_REVISIONS))
def test_two_target_alias_full_graph_add_repeat_scale_order_remove_recreates_exact_owner(graph_runtime, tmp_path, monkeypatch, profile_id):
    from modiff.native_disk_lora import disk_lora_binding_matches

    server, loaders, manager, _, pipelines = graph_runtime
    profile, revision = configure_alias_runtime(graph_runtime, monkeypatch, profile_id)
    graph = alias_graph(profile, revision)
    assert execute_graph(server, graph) == [[1., 2.]]
    assert_alias_outputs(server, [[1., 2.]])
    initial = weakref.ref(server.node_cache['models'].loader)
    assert execute_graph(server, graph) == [[1., 2.]]
    graph['nodes']['generate']['params']['prompt']['value'] = 'prompt-only warm'
    assert execute_graph(server, graph) == [[1., 2.]]
    assert server.node_cache['models'].loader is initial() and len(pipelines) == 1

    original_factory = loaders._instantiate_reviewed_builtin_pipeline
    previous = [initial]
    def require_old_owner_destroyed(*args, **kwargs):
        assert previous[0]() is None, 'Prior adapted owner survives replacement allocation'
        return original_factory(*args, **kwargs)
    monkeypatch.setattr(loaders, '_instantiate_reviewed_builtin_pipeline', require_old_owner_destroyed)

    changes = (
        ([descriptor(tmp_path)], [[4., 5.]]),
        ([descriptor(tmp_path, scale=.5)], [[2.5, 3.5]]),
        ([descriptor(tmp_path, name='one'), descriptor(tmp_path, 2., name='two')], [[16., 17.]]),
    )
    for loras, expected in changes:
        graph['nodes']['models']['params']['lora_list']['value'] = loras
        assert execute_graph(server, graph) == expected
        assert_alias_outputs(server, expected)
        node = server.node_cache['models']
        policy = node._native_disk_lora_state['policy']
        assert policy == (('transformer', 'text_encoder'), tuple(item['descriptor_sha256'] for item in loras))
        assert disk_lora_binding_matches(node, policy, manager)
        previous[0] = weakref.ref(node.loader)
        del node
        assert execute_graph(server, graph) == expected
        assert server.node_cache['models'].loader is previous[0]()

    graph['nodes']['models']['params']['lora_list']['value'].reverse()
    assert execute_graph(server, graph) == [[16., 17.]]
    assert_alias_outputs(server, [[16., 17.]])
    policy = server.node_cache['models']._native_disk_lora_state['policy']
    assert policy[1] == tuple(item['descriptor_sha256'] for item in graph['nodes']['models']['params']['lora_list']['value'])
    previous[0] = weakref.ref(server.node_cache['models'].loader)
    graph['nodes']['models']['params']['lora_list']['value'] = None
    assert execute_graph(server, graph) == [[1., 2.]]
    assert_alias_outputs(server, [[1., 2.]])
    assert server.node_cache['models']._native_disk_lora_state['policy'] == (('transformer', 'text_encoder'), ())


@pytest.mark.parametrize('profile_id', tuple(ALIAS_REVISIONS))
@pytest.mark.parametrize('change', ['invalidate', 'transformer_eviction', 'text_encoder_eviction', 'implementation'])
def test_two_target_alias_incomplete_state_releases_before_reallocation(graph_runtime, tmp_path, monkeypatch, profile_id, change):
    server, loaders, manager, _, _ = graph_runtime
    profile, revision = configure_alias_runtime(graph_runtime, monkeypatch, profile_id)
    graph = alias_graph(profile, revision, [descriptor(tmp_path)])
    assert execute_graph(server, graph) == [[4., 5.]]
    previous = weakref.ref(server.node_cache['models'].loader)
    if change == 'invalidate':
        server.node_cache['models'].invalidate_cache()
    elif change == 'implementation':
        original = loaders.ModelsLoader.execute
        def replacement(self, **kwargs):
            return original(self, **kwargs)
        monkeypatch.setattr(loaders.ModelsLoader, 'execute', replacement)
    else:
        name = change.removesuffix('_eviction')
        component_id = server.node_cache['models'].output['pipeline_components'][name]['model_id']
        manager.components.pop(component_id)
    original_factory = loaders._instantiate_reviewed_builtin_pipeline
    def require_destroyed(*args, **kwargs):
        assert previous() is None
        return original_factory(*args, **kwargs)
    monkeypatch.setattr(loaders, '_instantiate_reviewed_builtin_pipeline', require_destroyed)
    assert execute_graph(server, graph) == [[4., 5.]]
    assert_alias_outputs(server, [[4., 5.]])


@pytest.mark.parametrize('profile_id', tuple(ALIAS_REVISIONS))
@pytest.mark.parametrize('name', ['transformer', 'text_encoder'])
def test_two_target_alias_shared_component_refuses_before_adapter_mutation(graph_runtime, tmp_path, monkeypatch, profile_id, name):
    server, _, manager, events, _ = graph_runtime
    profile, revision = configure_alias_runtime(graph_runtime, monkeypatch, profile_id)
    graph = alias_graph(profile, revision, [descriptor(tmp_path)])
    assert execute_graph(server, graph) == [[4., 5.]]
    target = getattr(server.node_cache['models'].loader, name)
    ids = {key for key, value in manager.components.items() if value is target}
    manager.collections['other-owner'] = set(ids)
    before = len([event for event in events if event[0] == 'lora'])
    graph['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, scale=.5)]
    with pytest.raises(RuntimeError, match='exclusively owned|shared'):
        execute_graph(server, graph)
    assert len([event for event in events if event[0] == 'lora']) == before
    assert ids.issubset(manager.collections['other-owner'])


@pytest.mark.parametrize('profile_id', tuple(ALIAS_REVISIONS))
def test_two_target_alias_failed_injection_never_publishes_reusable_state(graph_runtime, tmp_path, monkeypatch, profile_id):
    server, _, _, _, _ = graph_runtime
    profile, revision = configure_alias_runtime(graph_runtime, monkeypatch, profile_id)
    original = TinyTwoTargetPipeline.load_lora_weights
    def fail_after_both_targets(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError('controlled two-target failure')
    monkeypatch.setattr(TinyTwoTargetPipeline, 'load_lora_weights', fail_after_both_targets)
    graph = alias_graph(profile, revision, [descriptor(tmp_path)])
    with pytest.raises(RuntimeError, match='controlled two-target failure'):
        execute_graph(server, graph)
    assert server.node_cache['models']._native_disk_lora_state is None
    previous = weakref.ref(server.node_cache['models'].loader)
    monkeypatch.setattr(TinyTwoTargetPipeline, 'load_lora_weights', original)
    assert execute_graph(server, graph) == [[4., 5.]] and previous() is None


@pytest.mark.parametrize('profile_id', tuple(ALIAS_REVISIONS))
@pytest.mark.parametrize('mismatch', ['installed_targets', 'missing_component'])
def test_two_target_alias_target_drift_refuses_before_weight_or_adapter_load(graph_runtime, tmp_path, monkeypatch, profile_id, mismatch):
    server, _, _, events, _ = graph_runtime
    profile, revision = configure_alias_runtime(graph_runtime, monkeypatch, profile_id)
    if mismatch == 'installed_targets':
        monkeypatch.setattr(TinyTwoTargetPipeline, '_lora_loadable_modules', ['transformer'])
    else:
        monkeypatch.setattr(TinyTwoTargetPipeline, 'pretrained_component_names', ['transformer', 'vae'])
    with pytest.raises(RuntimeError, match='reviewed installed pipeline|every declared target'):
        execute_graph(server, alias_graph(profile, revision, [descriptor(tmp_path)]))
    assert events == []
