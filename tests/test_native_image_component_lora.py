"""Actual native loader ownership contracts using tiny official CPU components.

Readiness/package and checker-ownership observations are explicit unit fixtures.
No safety algorithm, media forward, trained-model compatibility or hardware fit
is established here. Production seams import normally under tests/conftest.py.
"""
from copy import deepcopy
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
from diffusers import ComponentsManager, Cosmos3OmniTransformer, ComponentSpec, AutoencoderKLWan, UniPCMultistepScheduler
from transformers import Qwen2TokenizerFast
from safetensors.torch import save_file

from modiff.auxiliary_lora import build_lora_descriptor
from modiff.studio_execution_specs import studio_execution_spec_for_pair
from modiff.model_artifact_catalog import require_catalog_revision

MODEL = 'Cosmos3OmniModularPipeline'
REPO = 'nvidia/Cosmos3-Super-Text2Image'
REV = 'daf3d374804be4c512c2135568a7cb95d4341d79'


def graph(descriptors=None):
    specification = studio_execution_spec_for_pair(MODEL, 'text_to_image')
    params = dict(model_type=MODEL, repo_id={'source': 'hub', 'value': REPO}, revision=REV,
                  workflow_id='text2image', reviewed_variant=None, dtype='float32', device='cpu',
                  offload_mode='none', auto_offload=False, weight_loading_mode='cpu', lora_list=descriptors,
                  unet=None, vae=None, controlnet=None, quant_config=None, trust_remote_code=False)
    nodes = {'models': {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader',
                       'params': {key: {'value': value} for key, value in params.items()}}}
    for role, key, *_ in specification['roles']:
        if role == 'models':
            continue
        module, action = key.rsplit('.', 1)
        values = {} if role == 'preview' else dict(pipeline_class=MODEL, workflow_id='text2image',
                                                  block_path={'prompt': 'text_encoder', 'denoise': 'denoise', 'decode': 'decode'}[role])
        if role == 'prompt':
            values.update(prompt='synthetic contract only', negative_prompt='', num_frames=1, width=8, height=8)
        nodes[role] = {'module': module, 'action': action, 'params': {k: {'value': v} for k, v in values.items()}}
    for source, output, target, field in specification['edges']:
        nodes[target]['params'][field] = {'sourceId': source, 'sourceKey': output}
    return {'sid': 'unit', 'nodes': nodes, 'paths': [['models', 'prompt', 'denoise', 'decode', 'preview']]}


def descriptor(tmp_path, *, name='style', scale=1., prefix='transformer', shape=(1, 8), scheduler=''):
    directory = tmp_path / (name + '-' + prefix.replace('.', '-') + '-' + str(shape[1]))
    directory.mkdir(exist_ok=True)
    filename = directory / 'adapter.safetensors'
    save_file({prefix + '.proj_out.lora_A.weight': torch.ones(shape),
               prefix + '.proj_out.lora_B.weight': torch.ones((1, shape[0]))}, filename)
    return build_lora_descriptor(selection={'source': 'local', 'value': str(directory)},
                                 weight_name=filename.name, adapter_name=name, scale=scale,
                                 revision=None, expected_sha256=None, scheduler_class=scheduler,
                                 scheduler_config=None)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from modiff.config import CONFIG
    for field in ('work_dir', 'data', 'images', 'videos', 'audio', 'models', 'upscalers', 'temp'):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(tmp_path / 'hub'))
    monkeypatch.setitem(CONFIG.hf, 'online_status', 'Offline')
    monkeypatch.delenv('MODIFF_SUPERVISOR_QUEUE_STATE', raising=False)
    import modules
    import modules.ModularDiffusers as modular
    from modules.ModularDiffusers import loaders
    from modules.ModularDiffusers import cosmos_safety
    import modiff.optional_runtime_execution as readiness
    from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES
    from modiff.cosmos_safety_contract import cosmos_safety_artifacts
    from modiff.server import WebServer, memory_manager

    manager, events, references, observations = ComponentsManager(), [], [], []
    monkeypatch.setattr(modular, 'components', manager)
    monkeypatch.setattr(loaders, 'components', manager)
    monkeypatch.setattr(memory_manager, 'cache', {})
    monkeypatch.setitem(CONFIG.hf, 'online_status', 'Offline')
    cache = Path(CONFIG.hf['cache_dir'])
    snapshot = cache / 'models--nvidia--Cosmos3-Super-Text2Image' / 'snapshots' / REV
    snapshot.mkdir(parents=True, exist_ok=True)
    original_index = Path(__file__).resolve().parents[1] / 'tests/fixtures/cosmos3_super_t2i_modular_index.v1.json'
    (snapshot / 'modular_model_index.json').write_bytes(original_index.read_bytes())

    # ONLY observation data is simulated. Actual resolver/guard predicates run.
    ready = {'value': True}
    monkeypatch.setenv('MODIFF_RUNTIME_OVERLAY_STATUS', 'active')
    def observed_catalog():
        profile = OPTIONAL_RUNTIME_PROFILES['cosmos-guardrail-0.3.1']
        observations.append(('readiness-observation', ready['value'], profile.spec_digest))
        return {'schemaVersion': 1, 'profiles': [{
            'schemaVersion': 1, 'id': profile.id, 'label': 'UNIT OBSERVATION ONLY',
            'specDigest': profile.spec_digest, 'contractState': 'qualified', 'status': 'present_unqualified',
            'overlayStatus': 'active' if ready['value'] else 'missing', 'cutoverReady': ready['value'],
            'installActionAvailable': False, 'activationAvailable': False}],
            'overlay': {'processLoadStatus': 'active'}}
    monkeypatch.setattr(readiness, 'public_optional_runtime_catalog', observed_catalog)
    original_version = cosmos_safety.metadata.version
    def observed_version(name):
        if name == 'cosmos-guardrail':
            observations.append(('package-version-observation', name, '0.3.1'))
            return '0.3.1'
        if name in OPTIONAL_RUNTIME_PROFILES['cosmos-guardrail-0.3.1'].incompatible_distributions:
            observations.append(('UNIT-incompatible-distribution-observation', name, 'absent'))
            raise cosmos_safety.metadata.PackageNotFoundError(name)
        return original_version(name)
    monkeypatch.setattr(cosmos_safety.metadata, 'version', observed_version)

    class UnitReadinessObservationChecker(torch.nn.Module):
        """Ownership fixture only: NO check methods or moderation/forward."""
        def __init__(self, owner, model):
            super().__init__()
            self._modiff_cosmos_owned_checker = True
            self._modiff_cosmos_owner = owner
            self._modiff_cosmos_model_type = model
            self._modiff_cosmos_artifacts = tuple((x['repository'], x['revision']) for x in cosmos_safety_artifacts())
    def checker(*, owner_node_id, model_type):
        cosmos_safety.require_cosmos_safety_runtime()
        events.append(('checker-ownership-fixture', owner_node_id))
        value = UnitReadinessObservationChecker(owner_node_id, model_type)
        references.append(('checker', weakref.ref(value)))
        return value
    monkeypatch.setattr(cosmos_safety, 'build_cosmos_safety_checker', checker)

    def synthetic_load(spec, **kwargs):
        assert spec.pretrained_model_name_or_path == REPO and spec.revision == REV
        assert kwargs.get('trust_remote_code') is False
        events.append(('synthetic-official-component', spec.name))
        if spec.name == 'transformer':
            value = Cosmos3OmniTransformer(hidden_size=8, intermediate_size=16, head_dim=4,
                num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1, latent_channel=1,
                latent_patch_size=1, patch_latent_dim=1, vocab_size=16, rope_axes_dim=(1, 0, 1),
                action_gen=False, sound_gen=False).eval()
            references.append(('transformer', weakref.ref(value)))
        elif spec.name == 'vae':
            value = AutoencoderKLWan(base_dim=2, decoder_base_dim=2, z_dim=1, dim_mult=[1],
                num_res_blocks=1, attn_scales=[], temperal_downsample=[], latents_mean=[0.], latents_std=[1.])
        elif spec.name == 'text_tokenizer':
            value = Qwen2TokenizerFast(vocab={'<|endoftext|>': 0, '<|vision_start|>': 1, 'a': 2}, merges=[])
        elif spec.name == 'scheduler':
            value = UniPCMultistepScheduler()
        else:
            raise AssertionError('Unreviewed fixture component was requested: ' + spec.name)
        assert type(value) is spec.type_hint
        value._diffusers_load_id = spec.load_id
        if isinstance(value, torch.nn.Module):
            assert sum(p.numel() * p.element_size() for p in value.parameters()) < 1024**2
        return value
    monkeypatch.setattr(ComponentSpec, 'load', synthetic_load)
    server = WebServer(modules=modules.MODULE_MAP, work_dir=str(tmp_path), data_dir=str(tmp_path / 'data'))
    monkeypatch.setattr(server, 'queue_message', lambda _value: None)
    # These are receipt/resource observation fixtures, never an admission pass.
    monkeypatch.setattr(server, '_runtime_fingerprint', lambda **_kwargs: {'fingerprint': 'unit-cpu-contract-only'})
    monkeypatch.setattr(server, '_gpu_process_snapshot', lambda: {'observation': 'unobserved-CPU-unit-diagnostic'})
    monkeypatch.setattr(server, '_record_auto_resource_success', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, '_record_optimization_observations', lambda *_args, **_kwargs: [])
    actual_dispatch = server.execute_node
    dispatch_overrides = {}
    class LoadOnlyCheckpoint(Exception):
        pass
    def dispatch(id, node, sid, **kwargs):
        if node['action'] not in ('ModelsLoader', 'Lora'):
            raise LoadOnlyCheckpoint('Intentionally stopped before ALL stage/safety/media forwards.')
        return actual_dispatch(id, node, sid, quiet=True,
                               param_overrides=dispatch_overrides if node['action'] == 'ModelsLoader' else None)
    monkeypatch.setattr(server, 'execute_node', dispatch)
    def run(source):
        server.current_task = {'task_id': 'cpu-unit', 'sid': 'unit', 'status': 'running', 'progress': 0}
        with pytest.raises(LoadOnlyCheckpoint):
            server._execute_graph(deepcopy(source))
        owner = next(key for key, node in source['nodes'].items() if node['action'] == 'ModelsLoader')
        return server.node_cache[owner]
    result = SimpleNamespace(server=server, loaders=loaders, manager=manager, events=events,
                             references=references, ready=ready, observations=observations, run=run)
    result.dispatch_overrides = dispatch_overrides
    result.actual_dispatch = actual_dispatch
    yield result
    server.node_cache.clear()
    memory_manager.cache.clear()
    manager.components.clear()
    manager.collections.clear()
    gc.collect()


def test_actual_not_ready_observation_refuses_before_component_construction(runtime):
    from modiff.optional_runtime_execution import OptionalRuntimeExecutionBlocked
    runtime.ready['value'] = False
    with pytest.raises(OptionalRuntimeExecutionBlocked):
        runtime.run(graph())
    assert not runtime.events


def test_context_free_actual_updater_remains_refused(runtime, tmp_path):
    import diffusers
    value = diffusers.Cosmos3OmniModularPipeline(workflow='text2image')
    with pytest.raises(ValueError, match='does not expose the reviewed Diffusers LoRA lifecycle'):
        runtime.loaders.update_lora_adapters(value, [descriptor(tmp_path)])


def test_current_source_selected_actual_dispatch_loads_real_component_adapter(runtime, tmp_path):
    node = runtime.run(graph([descriptor(tmp_path)]))
    assert type(node.loader.transformer) is Cosmos3OmniTransformer
    assert tuple(node.loader.transformer.peft_config) == ('style',)
    assert node._native_disk_lora_state['componentComplete'] is True
    assert node._selected_component_lora_context is None
    assert node.loader.requires_safety_checker is True


def test_actual_source_same_owner_warm_and_order_scale_remove(runtime, tmp_path):
    first = descriptor(tmp_path, name='first')
    second = descriptor(tmp_path, name='second', scale=.5)
    source = graph([first, second])
    runtime.run(source)
    initial = weakref.ref(runtime.server.node_cache['models'].loader)
    count = len(runtime.events)
    runtime.run(source)
    assert runtime.server.node_cache['models'].loader is initial() and len(runtime.events) == count
    source['nodes']['prompt']['params']['prompt']['value'] = 'prompt changes do not change owner'
    runtime.run(source)
    assert runtime.server.node_cache['models'].loader is initial()
    source['nodes']['models']['params']['lora_list']['value'] = [second, first]
    runtime.run(source)
    assert initial() is None
    assert runtime.server.node_cache['models'].loader.transformer.active_adapters() == ['second', 'first']
    source['nodes']['models']['params']['lora_list']['value'] = None
    node = runtime.run(source)
    assert not getattr(node.loader.transformer, 'peft_config', {})


@pytest.mark.parametrize('change', ['frame', 'pipeline', 'task', 'stage_path', 'edge', 'revision', 'unknown_consumer', 'foreign_owner'])
def test_selected_source_scope_negatives_before_loading(runtime, tmp_path, change):
    source = graph([descriptor(tmp_path)])
    if change == 'frame': source['nodes']['prompt']['params']['num_frames']['value'] = 2
    if change == 'pipeline': source['nodes']['denoise']['params']['pipeline_class']['value'] = 'Cosmos3DistilledModularPipeline'
    if change == 'task': source['nodes']['prompt']['params']['workflow_id']['value'] = 'text2video'
    if change == 'stage_path': source['nodes']['prompt']['params']['block_path']['value'] = 'denoise'
    if change == 'edge': source['nodes']['denoise']['params']['state_in']['sourceKey'] = 'wrong'
    if change == 'revision': source['nodes']['models']['params']['revision']['value'] = 'a' * 40
    if change == 'unknown_consumer':
        source['nodes']['foreign'] = {'module': 'modules.Text', 'action': 'String', 'params': {
            'component': {'sourceId': 'models', 'sourceKey': 'pipeline_components'}}}
        source['paths'][0].append('foreign')
    if change == 'foreign_owner':
        source['nodes']['other-models'] = deepcopy(source['nodes']['models'])
        source['nodes']['denoise']['params']['pipeline_components']['sourceId'] = 'other-models'
        source['paths'][0].insert(0, 'other-models')
    with pytest.raises(ValueError): runtime.run(source)
    assert not runtime.events


@pytest.mark.parametrize('change', ['prefix', 'shape', 'scheduler'])
def test_invalid_adapter_refused_before_official_mutation(runtime, tmp_path, change):
    item = descriptor(tmp_path, prefix='vae' if change == 'prefix' else 'transformer',
                      shape=(1, 9) if change == 'shape' else (1, 8),
                      scheduler='FlowMatchEulerDiscreteScheduler' if change == 'scheduler' else '')
    with pytest.raises((ValueError, RuntimeError)): runtime.run(graph([item]))
    assert all(not getattr(reference(), 'peft_config', {}) for kind, reference in runtime.references
               if kind == 'transformer' and reference() is not None)


@pytest.mark.parametrize('change', ['component_eviction', 'output', 'inner_output', 'state_incomplete', 'scale_mutation', 'method'])
def test_exact_owner_incarnation_and_completion_changes_reconstruct(runtime, tmp_path, monkeypatch, change):
    source = graph([descriptor(tmp_path)])
    runtime.run(source)
    node = runtime.server.node_cache['models']
    initial = weakref.ref(node.loader)
    if change == 'component_eviction':
        keys = [key for key, value in runtime.manager.components.items() if value is node.loader.transformer]
        for key in keys: runtime.manager.components.pop(key)
    if change == 'output': node.output = deepcopy(node.output)
    if change == 'inner_output': node.output['pipeline_components']['transformer']['model_id'] = 'foreign-component'
    if change == 'state_incomplete': node._native_disk_lora_state['componentComplete'] = False
    if change == 'scale_mutation': node.loader.transformer.proj_out.scaling['style'] = .25
    if change == 'method':
        original = Cosmos3OmniTransformer.set_adapters
        def wrapped(self, *args, **kwargs): return original(self, *args, **kwargs)
        monkeypatch.setattr(Cosmos3OmniTransformer, 'set_adapters', wrapped)
    node = None
    runtime.run(source)
    assert initial() is None


def test_shared_target_refuses_before_owner_destruction(runtime, tmp_path):
    source = graph([descriptor(tmp_path)])
    runtime.run(source)
    node = runtime.server.node_cache['models']
    ids = {key for key, value in runtime.manager.components.items() if value is node.loader.transformer}
    runtime.manager.collections['external-shared-owner'] = ids
    source['nodes']['models']['params']['lora_list']['value'] = None
    with pytest.raises(ValueError, match='shared by another owner'): runtime.run(source)
    assert runtime.server.node_cache['models'] is node


def test_actual_none_owner_reassignment_destroys_former_before_factory(runtime, tmp_path):
    source = graph([descriptor(tmp_path)])
    runtime.run(source)
    initial = weakref.ref(runtime.server.node_cache['models'].loader)
    renamed = deepcopy(source)
    renamed['nodes']['new-models'] = renamed['nodes'].pop('models')
    for node in renamed['nodes'].values():
        for param in node['params'].values():
            if param.get('sourceId') == 'models': param['sourceId'] = 'new-models'
    renamed['paths'][0][0] = 'new-models'
    runtime.run(renamed)
    assert initial() is None and 'models' not in runtime.server.node_cache


def test_invalid_shape_preserves_existing_complete_owner_before_release(runtime, tmp_path):
    source = graph([descriptor(tmp_path)])
    runtime.run(source)
    initial = runtime.server.node_cache['models']
    events = len(runtime.events)
    source['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, shape=(1, 9))]
    with pytest.raises(ValueError, match='target shape'): runtime.run(source)
    assert runtime.server.node_cache['models'] is initial and len(runtime.events) == events


@pytest.mark.parametrize('failure', ['injection', 'activation', 'cancel_after_injection', 'safety_publication'])
def test_partial_failure_poison_and_retry_recreate_exact_owner(runtime, tmp_path, monkeypatch, failure):
    source = graph([descriptor(tmp_path)])
    with monkeypatch.context() as fault:
        if failure in {'injection', 'cancel_after_injection'}:
            original = Cosmos3OmniTransformer.load_lora_adapter
            def wrapped(self, *args, **kwargs):
                original(self, *args, **kwargs)
                if failure == 'injection':
                    raise RuntimeError('UNIT injected post-PEFT failure')
                runtime.server.node_cache['models']._interrupt = True
            fault.setattr(Cosmos3OmniTransformer, 'load_lora_adapter', wrapped)
        if failure == 'activation':
            original = Cosmos3OmniTransformer.set_adapters
            def wrapped(self, *args, **kwargs):
                original(self, *args, **kwargs)
                raise RuntimeError('UNIT injected post-activation failure')
            fault.setattr(Cosmos3OmniTransformer, 'set_adapters', wrapped)
        if failure == 'safety_publication':
            original = runtime.loaders.ModelsLoader.mm_add
            def wrapped(self, *args, **kwargs):
                original(self, *args, **kwargs)
                raise RuntimeError('UNIT injected post-MM checker registration failure')
            fault.setattr(runtime.loaders.ModelsLoader, 'mm_add', wrapped)
        with pytest.raises(RuntimeError): runtime.run(source)
    failed = runtime.server.node_cache['models']
    assert failed._native_disk_lora_state['componentComplete'] is False
    assert failed._cache_valid is False and failed._selected_component_lora_context is None
    previous = weakref.ref(failed.loader)
    failed = None
    runtime.run(source)
    assert previous() is None and runtime.server.node_cache['models']._native_disk_lora_state['componentComplete'] is True


@pytest.mark.parametrize('external', ['target', 'checker'])
def test_external_reference_stops_before_replacement_factory(runtime, tmp_path, external):
    from utils.memory_menager import memory_manager
    source = graph([descriptor(tmp_path)])
    runtime.run(source)
    node = runtime.server.node_cache['models']
    held = node.loader.transformer if external == 'target' else memory_manager.cache[node._mm_models[0]]['model']
    events = len(runtime.events)
    node = None
    source['nodes']['models']['params']['lora_list']['value'] = None
    with pytest.raises(ValueError, match='references remain alive'): runtime.run(source)
    assert len(runtime.events) == events and held is not None


def test_private_authority_tuple_cannot_be_injected_by_graph(runtime):
    source = graph()
    source['nodes']['models']['params']['_reviewed_component_lora_identity'] = {'value': ('forged',)}
    with pytest.raises(ValueError, match='cannot be supplied'): runtime.run(source)
    assert not runtime.events


@pytest.mark.parametrize('artifact_repository', [REPO, 'nvidia/Cosmos3-Nano'])
def test_actual_reviewed_variant_uses_super_artifact_and_separate_nano_graph_profile(
        runtime, tmp_path, artifact_repository):
    source = graph([descriptor(tmp_path)])
    owner = source['nodes']['models']['params']
    # Use the actual declared loader arguments; profile authority is resolved
    # from the immutable artifact, never a fictional executable node field.
    from inspect import signature
    assert 'execution_profile_id' not in runtime.loaders.ModelsLoader.params
    assert 'execution_profile_id' not in owner
    assert set(owner) <= set(signature(runtime.loaders.ModelsLoader.execute).parameters)
    owner['repo_id']['value'] = {'source': 'hub', 'value': artifact_repository}
    owner['revision']['value'] = require_catalog_revision(artifact_repository, model_type=MODEL)
    owner['reviewed_variant']['value'] = REPO if artifact_repository != REPO else None
    # The component issuer selects the declared graph alias; the pair
    # resolver must not grant an arbitrary fallback for the Super ID.
    assert studio_execution_spec_for_pair(MODEL, 'text_to_image',
        execution_profile_id='cosmos3-super-text-to-image:official-modular-workflow') is None
    node = runtime.run(source)
    signature = node._native_disk_lora_state['componentContext']
    assert signature[2] == 'cosmos3-super-text-to-image:official-modular-workflow'
    assert signature[14] == 'cosmos3-nano:official-modular-workflow'
    assert signature[5:7] == (REPO, REV)


def test_actual_native_descriptor_chain_keeps_order_and_complete_owner(runtime, tmp_path):
    first = descriptor(tmp_path, name='first')
    second = descriptor(tmp_path, name='second')
    source = graph()
    for key, item, previous in (('lora-first', first, None), ('lora-second', second, 'lora-first')):
        artifact = item['artifact']
        values = dict(model={'source': 'local', 'value': artifact['root']}, weight_name=artifact['weight_name'],
                      expected_sha256=artifact['sha256'], adapter_name=item['adapter_name'], scale=item['scale'],
                      revision='', scheduler_class='', scheduler_config='{}')
        params = {name: {'value': value} for name, value in values.items()}
        params['previous_loras'] = {'value': None} if previous is None else {'sourceId': previous, 'sourceKey': 'lora'}
        source['nodes'][key] = {'module': 'modules.ModularDiffusers', 'action': 'Lora', 'params': params}
    source['nodes']['models']['params']['lora_list'] = {'sourceId': 'lora-second', 'sourceKey': 'lora'}
    source['paths'][0] = ['lora-first', 'lora-second', *source['paths'][0]]
    node = runtime.run(source)
    assert node.loader.transformer.active_adapters() == ['first', 'second']
    assert node._native_disk_lora_state['componentComplete'] is True


def test_conditional_public_permission_keeps_disk_targets_and_qualification_unchanged(runtime):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    from modiff.studio_execution_specs import studio_capability_definition
    profile = DIFFUSERS_EXECUTION_PROFILES['cosmos3-super-text-to-image:official-modular-workflow']
    public = profile.to_public_dict()
    assert 'component_lora_declaration' not in public
    assert public['adapterAuthoring'] == {
        'schemaVersion': 1, 'mediaKind': 'image', 'adapterNodeKey': 'modules.ModularDiffusers.Lora',
        'tasks': ['text_to_image'], 'maxOwnedAdapters': 32,
        'repositoryRevisions': [{'repo': REPO, 'revision': REV}],
        'requiredControls': {'offload_mode': 'none', 'auto_offload': False, 'weight_loading_mode': 'cpu',
                             'quant_config': None, 'unet': None, 'vae': None, 'controlnet': None,
                             'trust_remote_code': False},
    }
    assert profile.lora_loadable_components == () and profile.live_proof is False
    assert studio_capability_definition(MODEL, repository=REPO)['supportsLora'] is False


@pytest.mark.parametrize('override', ['adapter', 'device', 'dtype', 'repository', 'private_tuple'])
def test_resolved_override_cannot_destroy_or_replace_current_source_owner(runtime, tmp_path, override):
    source = graph([descriptor(tmp_path)])
    runtime.run(source)
    original = runtime.server.node_cache['models']
    events = len(runtime.events)
    if override == 'adapter': runtime.dispatch_overrides['lora_list'] = [descriptor(tmp_path, name='foreign')]
    if override == 'device': runtime.dispatch_overrides['device'] = 'cuda:0'
    if override == 'dtype': runtime.dispatch_overrides['dtype'] = 'bfloat16'
    if override == 'repository': runtime.dispatch_overrides['repo_id'] = {'source': 'hub', 'value': 'nvidia/Cosmos3-Nano'}
    if override == 'private_tuple': runtime.dispatch_overrides['_reviewed_component_lora_identity'] = ('forged',)
    with pytest.raises(ValueError): runtime.run(source)
    assert runtime.server.node_cache['models'] is original and len(runtime.events) == events


def test_existing_context_free_vanilla_checker_cannot_survive_selected_replacement(runtime, tmp_path):
    from utils.memory_menager import memory_manager
    runtime.actual_dispatch('models', graph()['nodes']['models'], 'unit', quiet=True)
    node = runtime.server.node_cache['models']
    assert node._native_disk_lora_state is None
    held = memory_manager.cache[node._mm_models[0]]['model']
    node = None
    events = len(runtime.events)
    with pytest.raises(ValueError, match='references remain alive'):
        runtime.run(graph([descriptor(tmp_path)]))
    assert len(runtime.events) == events and held is not None


def test_selected_cold_owner_does_not_adopt_an_unattested_ordinary_loader(runtime, tmp_path, monkeypatch):
    runtime.actual_dispatch('ordinary', graph()['nodes']['models'], 'unit', quiet=True)
    ordinary = runtime.server.node_cache['ordinary']
    former = weakref.ref(ordinary)
    assert ordinary._native_disk_lora_state is None
    ordinary = None
    def forbidden_adoption(*_args):
        raise AssertionError('A selected component owner must not enter ordinary adoption.')
    monkeypatch.setattr(runtime.server, '_adopt_reusable_loader_node', forbidden_adoption)
    node = runtime.run(graph([descriptor(tmp_path)]))
    assert former() is None and node._native_disk_lora_state['componentComplete'] is True


def test_attempt_preflight_refuses_invalid_artifact_before_any_cleanup(runtime, tmp_path, monkeypatch):
    source = graph([descriptor(tmp_path)])
    runtime.run(source)
    initial = runtime.server.node_cache['models']
    source['nodes']['models']['params']['lora_list']['value'] = [descriptor(tmp_path, prefix='vae')]
    def forbidden_inner(_graph):
        raise AssertionError('Attempt cleanup must not begin for a bad selected artifact.')
    monkeypatch.setattr(runtime.server, '_execute_graph_inner', forbidden_inner)
    with pytest.raises(ValueError, match='Foreign prefix'): runtime.run(source)
    assert runtime.server.node_cache['models'] is initial


def test_reviewed_index_change_reconstructs_before_weight_allocation(runtime, tmp_path):
    from modiff.config import CONFIG
    import json
    source = graph([descriptor(tmp_path)])
    runtime.run(source)
    previous = weakref.ref(runtime.server.node_cache['models'].loader)
    path = Path(CONFIG.hf['cache_dir']) / 'models--nvidia--Cosmos3-Super-Text2Image/snapshots' / REV / 'modular_model_index.json'
    value = json.loads(path.read_text())
    value['_diffusers_version'] = '0.41.0'
    path.write_text(json.dumps(value), encoding='utf-8')
    runtime.run(source)
    assert previous() is None


def test_partial_selected_target_is_ineligible_for_ordinary_vanilla_reuse(runtime, tmp_path, monkeypatch):
    from modiff.modular_runtime_policy import normalize_modular_runtime_policy
    source = graph([descriptor(tmp_path)])
    original = Cosmos3OmniTransformer.load_lora_adapter
    def failure(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError('UNIT post-injection failure')
    monkeypatch.setattr(Cosmos3OmniTransformer, 'load_lora_adapter', failure)
    with pytest.raises(RuntimeError): runtime.run(source)
    component = runtime.server.node_cache['models'].loader.transformer
    assert runtime.loaders.component_reuse_compatible(component, dtype=torch.float32,
        requested_quantization=None, offload_mode='none', device='cpu', runtime_policy=normalize_modular_runtime_policy()) is False


@pytest.mark.parametrize('mode,automatic', [
    ('none', False), ('none', True),
    ('model_cpu', False), ('model_cpu', True),
    ('sequential_cpu', False), ('sequential_cpu', True),
    ('group_cpu', False), ('group_cpu', True),
    ('group_disk', False), ('group_disk', True),
    (None, None),
])
def test_vanilla_supported_loader_controls_preserve_actual_no_adapter_dispatch(runtime, mode, automatic):
    """Ordinary Expert CPU loader controls, not GPU/offload qualification."""
    source = graph(None)
    params = source['nodes']['models']['params']
    if mode is None:
        params.pop('offload_mode')
        params.pop('auto_offload')
    else:
        params['offload_mode']['value'] = mode
        params['auto_offload']['value'] = automatic
    node = runtime.run(source)
    assert type(node.loader.transformer) is Cosmos3OmniTransformer
    assert not getattr(node.loader.transformer, 'peft_config', {})
    assert node.loader.requires_safety_checker is True
    assert sum(event[0] == 'synthetic-official-component' for event in runtime.events) == 4


@pytest.mark.parametrize('mode,automatic', [
    ('none', True), ('model_cpu', True), ('group_cpu', True), ('group_disk', True),
])
def test_unsupported_component_placement_lora_stays_refused(runtime, tmp_path, mode, automatic):
    source = graph([descriptor(tmp_path)])
    source['nodes']['models']['params']['offload_mode']['value'] = mode
    source['nodes']['models']['params']['auto_offload']['value'] = automatic
    with pytest.raises((ValueError, RuntimeError)):
        runtime.run(source)


@pytest.mark.parametrize('mode,automatic', [('model_cpu', True), ('none', True)])
def test_selected_none_adapter_owner_releases_before_legacy_vanilla_replacement(runtime, tmp_path, mode, automatic):
    source = graph([descriptor(tmp_path)])
    runtime.run(source)
    old = weakref.ref(runtime.server.node_cache['models'].loader)
    owner = source['nodes']['models']['params']
    owner['lora_list']['value'] = None
    owner['offload_mode']['value'] = mode
    owner['auto_offload']['value'] = automatic
    ordinary = runtime.run(source)
    assert old() is None
    assert not getattr(ordinary.loader.transformer, 'peft_config', {})
    assert ordinary._native_disk_lora_state is None
    ordinary = None
    owner['offload_mode']['value'] = 'none'
    owner['auto_offload']['value'] = False
    selected = runtime.run(source)
    assert selected._native_disk_lora_state['componentComplete'] is True
    assert not getattr(selected.loader.transformer, 'peft_config', {})


def test_formatted_one_frame_source_dispatch_and_declared_normalizer(runtime):
    """Actual loader dispatch plus existing scalar helper, NOT stage/safety execution."""
    from modules.ModularDiffusers.workflow_blocks import _bounded_integer
    source = graph(None)
    source['nodes']['prompt']['params']['num_frames']['value'] = '1'
    node = runtime.run(source)
    assert node.output['pipeline_components']
    assert _bounded_integer('1', label='Cosmos 3 Omni frame count', minimum=1, maximum=481) == 1


@pytest.mark.parametrize('value', [True, 1.1, '1.0', '01', ' 1', 0, 2])
def test_one_frame_source_controls_match_declared_normalizer(runtime, value):
    """No block/checker forward; exact T2I one-frame restriction retained."""
    from modules.ModularDiffusers.workflow_blocks import _bounded_integer
    source = graph(None)
    source['nodes']['prompt']['params']['num_frames']['value'] = value
    with pytest.raises(ValueError):
        runtime.run(source)
        normalized = _bounded_integer(value, label='Cosmos 3 Omni frame count', minimum=1, maximum=481)
        if normalized != 1:
            raise ValueError('Cosmos 3 Omni text-to-image requires exactly one frame.')


def test_current_optional_input_export_shape_accepts_dense_selected_adapter(runtime, tmp_path):
    """Current factory optional fields serialize as display-only; no implicit grant."""
    source = graph([descriptor(tmp_path)])
    for name in ('quant_config', 'unet', 'vae', 'controlnet'):
        definition = runtime.loaders.ModelsLoader.params[name]
        assert definition['display'] == 'input'
        assert definition.get('required') is not True
        assert 'value' not in definition and 'default' not in definition
        source['nodes']['models']['params'][name] = {'display': 'input'}
    node = runtime.run(source)
    assert tuple(node.loader.transformer.peft_config) == ('style',)
    assert node._native_disk_lora_state['componentComplete'] is True
    assert node.loader.requires_safety_checker is True
