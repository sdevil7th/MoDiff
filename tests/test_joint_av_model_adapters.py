"""Required adapter ownership tests; mocked file receipt is not artifact qualification."""
from dataclasses import replace
import gc
from types import SimpleNamespace
from unittest import mock
import weakref

import pytest
import torch
from peft import LoraConfig, get_peft_model

from modiff import auxiliary_lora
from modiff import joint_av_model_adapters as adapters


def selection(workflow='t2va'):
    return adapters.required_joint_av_adapter(adapters.PIKA_PIPELINE,
        adapters.PIKA_REPOSITORY, adapters.PIKA_REVISION, workflow)


class Component(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(4, 4))

    def active_adapters(self):
        return self.network.active_adapters


class Pipeline:
    def __init__(self, contract):
        self.contract = contract
        self.transformer = Component() if contract.component == 'transformer' else None
        self.transformer_ref = Component() if contract.component == 'transformer_ref' else None
        self.loads = []
        self.activations = []
        self.inventory = {}
        self._collection = 'pika-owner'
        self._components_manager = SimpleNamespace(
            components={'actual-component': self.component},
            collections={self._collection: {'actual-component'}},
        )

    def get_list_adapters(self):
        return self.inventory

    def get_active_adapters(self):
        return self.component.active_adapters()

    @property
    def component(self):
        return getattr(self, self.contract.component)

    def load_lora_weights(self, directory, **kwargs):
        self.loads.append((directory, kwargs))
        self.component.network = get_peft_model(self.component.network,
            LoraConfig(r=128, lora_alpha=128, target_modules=['0']), adapter_name=adapters.ADAPTER_NAME)
        self.inventory = {self.contract.component: [adapters.ADAPTER_NAME]}

    def set_adapters(self, names, weights):
        self.activations.append((names, weights))


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    monkeypatch.setattr(adapters, '_require_runtime', lambda pipeline: None)
    monkeypatch.setattr(adapters, '_validate_schedulers', lambda pipeline: None)
    contract = selection()
    resolved = auxiliary_lora.ResolvedLoraDescriptor('descriptor-digest', 'hub',
        adapters.PIKA_REPOSITORY, adapters.PIKA_REVISION, tmp_path, 'adapter.safetensors',
        contract.sha256, adapters.ADAPTER_NAME, 1.0, None, {}, contract.weight_name)
    monkeypatch.setattr(auxiliary_lora, 'resolve_lora_descriptor', lambda descriptor: resolved)
    return {}, resolved


def test_declaration_lookup_is_exact_and_does_not_grant_other_models_an_adapter():
    assert selection('t2va') == selection('fl2va')
    assert selection('ref2va').component == 'transformer_ref'
    assert selection('ref2va').sha256 != selection().sha256
    assert adapters.required_joint_av_adapter('MiniMaxH3ModularPipeline', None, None, None) is None
    for repository, revision, workflow in [
        ('other/model', adapters.PIKA_REVISION, 't2va'),
        (adapters.PIKA_REPOSITORY, 'main', 't2va'),
        (adapters.PIKA_REPOSITORY, adapters.PIKA_REVISION, 'other'),
    ]:
        with pytest.raises(ValueError):
            adapters.required_joint_av_adapter(adapters.PIKA_PIPELINE, repository, revision, workflow)


def test_apply_uses_official_file_api_once_and_keeps_fixed_scale(prepared):
    pipeline = Pipeline(selection())
    result = adapters.apply_required_model_adapter(pipeline, selection(), prepared)
    assert result == (prepared[1],)
    assert pipeline.loads == [(str(prepared[1].load_directory), dict(weight_name='adapter.safetensors',
        adapter_name=adapters.ADAPTER_NAME, use_safetensors=True, local_files_only=True,
        low_cpu_mem_usage=True, load_into_transformer_ref=False))]
    assert pipeline.activations == [([adapters.ADAPTER_NAME], [1.0])]
    adapters.require_applied_model_adapter(pipeline.component, selection())
    adapters.apply_required_model_adapter(pipeline, selection(), prepared)
    assert len(pipeline.loads) == len(pipeline.activations) == 1


def test_reference_adapter_routes_only_into_reference_partition(prepared, monkeypatch):
    contract = selection('ref2va')
    resolved = replace(prepared[1], content_sha256=contract.sha256, artifact_weight_name=contract.weight_name)
    monkeypatch.setattr(auxiliary_lora, 'resolve_lora_descriptor', lambda descriptor: resolved)
    pipeline = Pipeline(contract)
    adapters.apply_required_model_adapter(pipeline, contract, ({}, resolved))
    assert pipeline.loads[0][1]['load_into_transformer_ref'] is True
    adapters.require_applied_model_adapter(pipeline.component, contract)
    with pytest.raises(ValueError, match='binding'):
        adapters.require_applied_model_adapter(pipeline.component, selection())


@pytest.mark.parametrize('ownership', ['shared', 'foreign', 'unregistered', 'missing-manager'])
def test_first_adapter_injection_requires_exclusive_registered_owner(prepared, ownership):
    pipeline = Pipeline(selection())
    manager = pipeline._components_manager
    if ownership == 'shared':
        manager.collections['another-owner'] = {'actual-component'}
    elif ownership == 'foreign':
        pipeline._collection = 'another-owner'
    elif ownership == 'unregistered':
        manager.components.clear()
    else:
        pipeline._components_manager = None
    with pytest.raises(ValueError, match='exclusively owned'):
        adapters.apply_required_model_adapter(pipeline, selection(), prepared)
    assert not pipeline.loads
    assert not pipeline.activations
    assert not hasattr(pipeline.component.network, 'peft_config')


def test_bound_component_reuse_does_not_mutate_a_later_shared_owner(prepared):
    pipeline = Pipeline(selection())
    adapters.apply_required_model_adapter(pipeline, selection(), prepared)
    pipeline._components_manager.collections['stage-owner'] = {'actual-component'}
    adapters.apply_required_model_adapter(pipeline, selection(), prepared)
    assert len(pipeline.loads) == len(pipeline.activations) == 1


def test_missing_foreign_and_changed_adapter_state_fail_closed(prepared):
    pipeline = Pipeline(selection())
    with pytest.raises(ValueError, match='binding'):
        adapters.require_applied_model_adapter(pipeline.component, selection())
    pipeline.inventory = {'transformer': ['foreign']}
    with pytest.raises(ValueError, match='foreign'):
        adapters.apply_required_model_adapter(pipeline, selection(), prepared)
    assert not pipeline.loads
    pipeline.inventory = {}
    adapters.apply_required_model_adapter(pipeline, selection(), prepared)
    from peft.tuners.tuners_utils import BaseTunerLayer
    layer = next(layer for layer in pipeline.component.modules() if isinstance(layer, BaseTunerLayer))
    layer.scaling[adapters.ADAPTER_NAME] = 0.0
    with pytest.raises(ValueError, match='scale one'):
        adapters.require_applied_model_adapter(pipeline.component, selection())
    with pytest.raises(ValueError, match='scale one'):
        adapters.apply_required_model_adapter(pipeline, selection(), prepared)


def test_changed_file_receipt_cannot_inject_or_borrow_another_component(prepared, monkeypatch):
    pipeline = Pipeline(selection())
    monkeypatch.setattr(auxiliary_lora, 'resolve_lora_descriptor',
                        lambda descriptor: replace(prepared[1], content_sha256='0' * 64))
    with pytest.raises(ValueError, match='changed'):
        adapters.apply_required_model_adapter(pipeline, selection(), prepared)
    assert not pipeline.loads


def test_weak_receipt_does_not_keep_retired_model_component_alive(prepared):
    pipeline = Pipeline(selection())
    adapters.apply_required_model_adapter(pipeline, selection(), prepared)
    component = weakref.ref(pipeline.component)
    del pipeline
    gc.collect()
    assert component() is None


def test_preflight_uses_existing_bounded_immutable_descriptor_boundary(tmp_path, monkeypatch):
    artifact = tmp_path / 'adapter.safetensors'
    artifact.write_bytes(b'fixture')
    contract = selection()
    resolved = auxiliary_lora.ResolvedLoraDescriptor('fixture', 'hub', adapters.PIKA_REPOSITORY,
        adapters.PIKA_REVISION, tmp_path, artifact.name, contract.sha256, adapters.ADAPTER_NAME,
        1.0, None, {}, contract.weight_name)
    calls = []
    monkeypatch.setattr(auxiliary_lora, 'build_lora_descriptor', lambda **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(auxiliary_lora, 'resolve_lora_descriptor', lambda descriptor: resolved)
    with pytest.raises(ValueError, match='byte size'):
        adapters.prepare_required_model_adapter(contract)
    assert calls[0] == dict(selection={'source': 'hub', 'value': adapters.PIKA_REPOSITORY},
        weight_name=contract.weight_name, revision=adapters.PIKA_REVISION,
        expected_sha256=contract.sha256, adapter_name=adapters.ADAPTER_NAME,
        scale=1.0, maximum_weight_bytes=contract.byte_size)


def prepare_loader_preflight(monkeypatch):
    from modules.ModularDiffusers import loaders
    node = loaders.ModelsLoader('required-adapter-loader')
    monkeypatch.setattr(node, '_effective_builtin_selector', lambda **kwargs: (
        {'source': 'hub', 'value': adapters.PIKA_REPOSITORY}, adapters.PIKA_REVISION))
    monkeypatch.setattr(node, '_preflight_reviewed_builtin_selection', lambda **kwargs: (
        'hub', adapters.PIKA_REPOSITORY, adapters.PIKA_REVISION, 'modular_model_index.json', {}))
    monkeypatch.setattr(loaders, '_reviewed_builtin_workflow_id', lambda model, workflow: workflow)
    return node


def test_loader_revalidates_missing_required_bytes_before_any_cache_reuse(monkeypatch):
    from modiff.NodeBase import NodeBase
    node = prepare_loader_preflight(monkeypatch)
    monkeypatch.setattr(adapters, 'prepare_required_model_adapter',
                        mock.Mock(side_effect=FileNotFoundError('not installed')))
    with mock.patch.object(NodeBase, '__call__') as cached:
        with pytest.raises(FileNotFoundError, match='not installed'):
            node._call_checked(model_type=adapters.PIKA_PIPELINE, workflow_id='t2va')
    cached.assert_not_called()


def test_cached_loader_also_requires_actual_active_component_binding(prepared, monkeypatch):
    from modiff.NodeBase import NodeBase
    node = prepare_loader_preflight(monkeypatch)
    node.loader = Pipeline(selection())
    monkeypatch.setattr(adapters, 'prepare_required_model_adapter', lambda selected: prepared)
    with mock.patch.object(NodeBase, '__call__', return_value={'cached': True}) as cached:
        with pytest.raises(ValueError, match='binding'):
            node._call_checked(model_type=adapters.PIKA_PIPELINE, workflow_id='t2va')
        assert cached.call_args.kwargs['_required_model_adapter_identity'] == prepared[1].descriptor_sha256
        adapters.apply_required_model_adapter(node.loader, selection(), prepared)
        assert node._call_checked(model_type=adapters.PIKA_PIPELINE, workflow_id='t2va') == {'cached': True}


def test_loader_rejects_extra_style_adapters_before_loading_or_reusing(monkeypatch):
    from modiff.NodeBase import NodeBase
    node = prepare_loader_preflight(monkeypatch)
    with mock.patch.object(NodeBase, '__call__') as cached:
        with pytest.raises(ValueError, match='extra style adapters'):
            node._call_checked(model_type=adapters.PIKA_PIPELINE, workflow_id='t2va', lora_list=[{}])
    cached.assert_not_called()


@pytest.mark.parametrize('name', ['unet', 'vae', 'controlnet'])
@pytest.mark.parametrize('connected', [{}, object()])
def test_loader_rejects_component_overrides_before_adapter_resolution_or_cache(monkeypatch, name, connected):
    from modiff.NodeBase import NodeBase
    node = prepare_loader_preflight(monkeypatch)
    prepare = mock.Mock()
    monkeypatch.setattr(adapters, 'prepare_required_model_adapter', prepare)
    with mock.patch.object(NodeBase, '__call__') as cached:
        with pytest.raises(ValueError, match='load its own components'):
            node._call_checked(model_type=adapters.PIKA_PIPELINE, workflow_id='t2va', **{name: connected})
    prepare.assert_not_called()
    cached.assert_not_called()


@pytest.mark.parametrize('name', ['transformer', 'transformer_ref'])
def test_required_adapter_identity_separates_implicit_component_reuse(name):
    from modules.ModularDiffusers.loaders import (
        _component_runtime_policy, assert_explicit_component_runtime_policy, reusable_component_ids,
    )
    binding = {'repository': adapters.PIKA_REPOSITORY, 'revision': adapters.PIKA_REVISION,
               'weight_name': selection().weight_name, 'sha256': selection().sha256,
               'component': selection().component, 'scale': 1.0}
    request = {'required_model_adapter': binding}
    component = SimpleNamespace(_modiff_modular_runtime_policy={})
    with pytest.raises(ValueError, match='policy'):
        assert_explicit_component_runtime_policy(name, component, request)
    stored = _component_runtime_policy(name, request)
    assert stored['required_model_adapter'] == binding
    unadapted, adapted = Component(), Component()
    adapted._modiff_modular_runtime_policy = stored
    manager = SimpleNamespace(_lookup_ids=lambda **_: ['unadapted', 'adapted'],
        get_one=lambda component_id: {'unadapted': unadapted, 'adapted': adapted}[component_id])
    assert reusable_component_ids(manager, name=name, load_id='same-pika-base', dtype=torch.float32,
        requested_quantization=None, offload_mode='none', device='cpu', runtime_policy=request) == ['adapted']
    binding['scale'] = 0
    assert stored['required_model_adapter']['scale'] == 1.0


def test_stage_refuses_missing_adapter_before_creating_an_official_block(monkeypatch):
    from modules.ModularDiffusers import workflow_blocks
    token = SimpleNamespace(_repo_id=adapters.PIKA_REPOSITORY, _revision=adapters.PIKA_REVISION)
    monkeypatch.setattr(workflow_blocks, 'require_component_binding', lambda *args, **kwargs: token)
    monkeypatch.setattr(workflow_blocks, 'collect_model_ids', lambda *args, **kwargs: ['actual-component'])
    monkeypatch.setattr(workflow_blocks, 'components', SimpleNamespace(
        get_components_by_ids=lambda **kwargs: {'transformer': Component()}))
    node = workflow_blocks.WorkflowJointAVTextEncode('required-stage')
    with mock.patch.object(workflow_blocks, 'pipeline_class_from_model_type') as create:
        with pytest.raises(ValueError, match='binding'):
            node._prepare_pipeline(pipeline_components={}, pipeline_class=adapters.PIKA_PIPELINE,
                                   workflow_id='t2va', block_path='text_encoder')
    create.assert_not_called()


@pytest.mark.parametrize('workflow', ['t2va', 'ref2va'])
def test_installed_reviewed_pipeline_loads_real_tiny_peft_file_with_normal_boundaries(tmp_path, monkeypatch, workflow):
    """A generated tiny fixture tests the actual API; it is not the published adapter."""
    from modiff.runtime_overlays import active_diffusers_provider
    provider = active_diffusers_provider()
    if not provider or provider.get('profileId') != 'huggingface-diffusers-video-d0b9b1b':
        pytest.skip('Requires the installed and validated reviewed video provider')
    import hashlib
    from diffusers import MiniMaxH3Transformer3DModel, MiniMaxH3Scheduler, PikaH3FlexBlocks
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from utils import huggingface
    config = dict(num_attention_heads=2, attention_head_dim=16, hidden_size=24,
        num_layers=2, num_refiner_layers=2, ffn_dim=32, in_channels=4, audio_in_channels=6,
        patch_size=(1, 2, 2), text_dim=8, freq_dim=8, time_embed_hidden_dim=24,
        time_embed_dim=16, rope_freq_dim=2)
    trained = MiniMaxH3Transformer3DModel(**config)
    trained.add_adapter(LoraConfig(r=128, lora_alpha=128, target_modules=['to_q']),
                        adapter_name=adapters.ADAPTER_NAME)
    contract = selection(workflow)
    cache = tmp_path / 'hub'
    snapshot = cache / ('models--' + adapters.PIKA_REPOSITORY.replace('/', '--')) / 'snapshots' / adapters.PIKA_REVISION
    artifact = snapshot / contract.weight_name
    artifact.parent.mkdir(parents=True)
    # The pinned Pika header uses PEFT's .default. infix without a component
    # prefix. The official H3 loader recognizes this exact published layout.
    state = get_peft_model_state_dict(trained, adapter_name=adapters.ADAPTER_NAME)
    state = {name.replace('.lora_A.weight', '.lora_A.default.weight')
                 .replace('.lora_B.weight', '.lora_B.default.weight'): value for name, value in state.items()}
    save_file(state, str(artifact))
    fixture = replace(contract, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                      byte_size=artifact.stat().st_size)
    monkeypatch.setattr(adapters, '_REFERENCE_ADAPTER' if workflow == 'ref2va' else '_FRAME_ADAPTER', fixture)
    monkeypatch.setattr(huggingface, 'cached_file_path', lambda *args, **kwargs: str(artifact))
    monkeypatch.setattr(huggingface, '_hf_cache_locations', lambda: [('fixture', str(cache))])
    prepared = adapters.prepare_required_model_adapter(fixture)
    from diffusers import ComponentsManager
    pipeline = PikaH3FlexBlocks().get_workflow(workflow).init_pipeline(
        components_manager=ComponentsManager(), collection='fixture-owner')
    pipeline.update_components(**{fixture.component: MiniMaxH3Transformer3DModel(**config),
        'scheduler': MiniMaxH3Scheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True),
        'audio_scheduler': MiniMaxH3Scheduler(shift=3.0, stochastic_sampling=True, shift_provided_sigmas=True)})
    adapters.apply_required_model_adapter(pipeline, fixture, prepared)
    component = getattr(pipeline, fixture.component)
    adapters.require_applied_model_adapter(component, fixture)
    assert pipeline.get_list_adapters()[fixture.component] == [adapters.ADAPTER_NAME]
    adapters.apply_required_model_adapter(pipeline, fixture, prepared)
