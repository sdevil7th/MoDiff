from types import SimpleNamespace

import pytest
import torch

from modules.ModularDiffusers import loaders


class PolicyModule(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.calls = []
        self.use_slicing = False
        self.use_tiling = True

    def set_attention_backend(self, backend):
        self.calls.append(('attention', backend))

    def reset_attention_backend(self):
        self.calls.append(('reset',))

    def enable_slicing(self):
        self.calls.append(('slicing', True))
        self.use_slicing = True

    def disable_slicing(self):
        self.calls.append(('slicing', False))
        self.use_slicing = False

    def enable_tiling(self):
        self.calls.append(('tiling', True))
        self.use_tiling = True

    def disable_tiling(self):
        self.calls.append(('tiling', False))
        self.use_tiling = False


class VisionPolicyModule(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.config = SimpleNamespace(vision_config=SimpleNamespace(_attn_implementation='sdpa'),
                                      text_config=SimpleNamespace(_attn_implementation='sdpa'))
        self.vision_calls = []

    def set_attn_implementation(self, value):
        self.vision_calls.append(value)
        self.config.vision_config._attn_implementation = value['vision_config']


def pipeline():
    transformer, vae, encoder = PolicyModule(), PolicyModule(), torch.nn.Linear(2, 2)
    return SimpleNamespace(transformer=transformer, vae=vae, text_encoder=encoder,
                           components={'transformer': transformer, 'vae': vae, 'text_encoder': encoder})


def test_saved_native_nodes_keep_unset_policy_and_do_not_call_hooks():
    assert loaders.ModelsLoader.params['attention_backend']['default'] == 'inherit'
    assert loaders.ModelsLoader.params['vae_slicing']['default'] is None
    assert loaders.ModelsLoader.params['vae_tiling']['default'] is None
    assert 'attention_backend' not in loaders.AutoModelLoader.params
    owner = pipeline()
    result = loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy())
    assert result['attention'] is result['vae'] is None
    assert owner.transformer.calls == owner.vae.calls == []
    assert not owner.vae.use_slicing and owner.vae.use_tiling


def test_explicit_template_policy_uses_existing_attention_and_vae_helpers():
    owner = pipeline()
    result = loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy('_native_math', True, True))
    assert owner.transformer.calls == [('attention', '_native_math')]
    assert owner.vae.calls == [('slicing', True), ('tiling', True)]
    assert result['attention']['applied'] == ['transformer']
    assert result['vae']['unsupported'] == []
    assert result['vae']['applied'] == [{'feature': 'slicing', 'enabled': True}, {'feature': 'tiling', 'enabled': True}]


def test_one_explicit_vae_feature_never_changes_the_unset_other_feature():
    owner = pipeline()
    result = loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy(vae_slicing=False))
    assert owner.vae.calls == [('slicing', False)]
    assert owner.vae.use_tiling
    assert result['vae'] == {'applied': [{'feature': 'slicing', 'enabled': False}], 'unsupported': []}


@pytest.mark.parametrize('kwargs,error', [
    ({'attention_backend': 'flash'}, ValueError), ({'attention_backend': False}, TypeError),
    ({'vae_slicing': 'true'}, TypeError), ({'vae_tiling': 1}, TypeError),
])
def test_unsupported_policy_fails_before_repository_or_model_loading(monkeypatch, kwargs, error):
    node = loaders.ModelsLoader('bad-policy')
    monkeypatch.setattr(node, '_effective_builtin_selector', lambda **_: pytest.fail('policy was not preflighted'))
    with pytest.raises(error):
        node.execute('TestPipeline', {'source': 'hub', 'value': 'test/model'}, 'cpu', torch.float32, **kwargs)


def compatible(model, policy=None):
    return loaders.component_reuse_compatible(model, dtype=torch.float32, requested_quantization=None,
        offload_mode='none', device='cpu', runtime_policy=policy)


def test_component_reuse_is_scoped_to_relevant_attention_and_vae_policies():
    owner = pipeline()
    policy = loaders.modular_runtime_policy('_native_math', True, True)
    loaders.apply_modular_runtime_policy(owner, policy)
    loaders.record_pipeline_component_runtime_policy(owner, offload_mode='none', device='cpu', runtime_policy=policy)
    assert compatible(owner.transformer, {'attention_backend': '_native_math'})
    assert not compatible(owner.transformer)
    assert not compatible(owner.transformer, {'attention_backend': 'auto'})
    assert compatible(owner.vae, {'vae_slicing': True, 'vae_tiling': True})
    assert not compatible(owner.vae, {'vae_slicing': True})
    assert compatible(owner.text_encoder)  # Unrelated weights need no reload.
    loaders.assert_explicit_component_runtime_policy('transformer', owner.transformer, policy)
    before = list(owner.transformer.calls)
    with pytest.raises(ValueError, match='different attention/VAE policy'):
        loaders.assert_explicit_component_runtime_policy('transformer', owner.transformer, loaders.modular_runtime_policy('auto'))
    assert owner.transformer.calls == before


def test_lookup_reuses_only_matching_policy_without_mutating_another_owner():
    inherited, configured = PolicyModule(), PolicyModule()
    configured._modiff_modular_runtime_policy = {'attention_backend': '_native_math'}
    manager = SimpleNamespace(_lookup_ids=lambda **_: ['inherited', 'configured'],
                              get_one=lambda component_id: {'inherited': inherited, 'configured': configured}[component_id])
    ids = loaders.reusable_component_ids(manager, name='transformer', load_id='same-weights', dtype=torch.float32,
        requested_quantization=None, offload_mode='none', device='cpu', runtime_policy=loaders.modular_runtime_policy('_native_math'))
    assert ids == ['configured']
    assert inherited.calls == configured.calls == []


def test_partial_configuration_failure_cannot_be_reused_as_untouched_legacy_policy():
    owner = pipeline()
    def fail(_backend):
        raise RuntimeError('driver kernel rejected')
    owner.transformer.set_attention_backend = fail
    with pytest.raises(RuntimeError, match='driver kernel rejected'):
        loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy('_native_math', True, True))
    assert not compatible(owner.transformer)
    assert not compatible(owner.vae)
    assert compatible(owner.text_encoder)


def test_existing_helpers_apply_math_attention_and_vae_flags_to_real_tiny_cpu_models():
    from diffusers import AutoencoderKL, UNet2DConditionModel

    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32,), down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
        cross_attention_dim=16, attention_head_dim=8, norm_num_groups=8)
    vae = AutoencoderKL(in_channels=3, out_channels=3, block_out_channels=(32,), latent_channels=4,
                        norm_num_groups=8, sample_size=8)
    owner = SimpleNamespace(unet=unet, vae=vae, components={'unet': unet, 'vae': vae})
    sample, conditioning = torch.randn(1, 4, 8, 8), torch.randn(1, 2, 16)
    with torch.no_grad():
        before = unet(sample, 1, conditioning).sample
    result = loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy('_native_math', True, True))
    with torch.no_grad():
        after = unet(sample, 1, conditioning).sample
    assert result['attention']['applied'] == ['unet']
    assert vae.use_slicing and vae.use_tiling
    assert torch.isfinite(after).all()
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)


@pytest.fixture
def dummy_native_execution(monkeypatch, request):
    """Exercise the normal ModelsLoader body with CPU modules, without weights."""
    entries, loads, instances = {}, [], []

    include_vision = bool(getattr(request, 'param', False))

    class DummyPipeline:
        pretrained_component_names = ['transformer', 'vae'] + (['text_encoder'] if include_vision else [])
        _component_specs = {}

        def __init__(self):
            self.components = {}

        def get_component_spec(self, name):
            return SimpleNamespace(load_id=f'exact-weight-id:{name}')

        def update_components(self, **values):
            self.components.update(values)
            for name, component in values.items():
                setattr(self, name, component)

    manager = SimpleNamespace(
        _lookup_ids=lambda name, **_: [key for key, (component_name, _) in entries.items() if component_name == name],
        get_one=lambda component_id: entries[component_id][1],
        get_components_by_ids=lambda ids, **_: {entries[key][0]: entries[key][1] for key in ids},
    )
    def instantiate(*_args, **_kwargs):
        instance = DummyPipeline()
        instances.append(instance)
        return instance
    def load(owner, *, names, diagnostics, **_kwargs):
        loads.append(list(names))
        for name in names:
            model = VisionPolicyModule() if name == 'text_encoder' else PolicyModule()
            entries[f'{name}:{len(entries)}'] = (name, model)
            owner.update_components(**{name: model})
            diagnostics.setdefault('components_loaded', []).append(name)
    monkeypatch.setattr(loaders, 'components', manager)
    monkeypatch.setattr(loaders, '_instantiate_reviewed_builtin_pipeline', instantiate)
    monkeypatch.setattr(loaders, '_reviewed_builtin_workflow_id', lambda *_: None)
    monkeypatch.setattr(loaders, '_reviewed_loader_component_outputs', lambda *_: ())
    monkeypatch.setattr(loaders, 'reviewed_modular_weight_variant', lambda *_: None)
    monkeypatch.setattr(loaders, '_loader_text_encoder_component_names', lambda *_: ['text_encoder'] if include_vision else [])
    monkeypatch.setattr(loaders, 'pin_modular_component_revisions', lambda *_: {})
    monkeypatch.setattr(loaders, '_primary_component_cache_dirs', lambda *_: {})
    monkeypatch.setattr(loaders, 'configure_components_manager_offload', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loaders, 'load_components_strict', load)
    monkeypatch.setattr(loaders, 'node_get_component_info', lambda **kwargs: {
        'model_id': next(key for key, (name, model) in entries.items()
                         if name == kwargs['name'] and model is getattr(instances[-1], name)),
    })

    def run(**policy):
        node = loaders.ModelsLoader(f'owner:{len(instances)}')
        monkeypatch.setattr(node, '_effective_builtin_selector', lambda **kwargs: (kwargs['repo_id'], 'a' * 40))
        monkeypatch.setattr(node, '_preflight_reviewed_builtin_selection', lambda **_: (
            'hub', 'test/model', 'a' * 40, 'model_index.json', {'_class_name': 'TestPipeline'},
        ))
        output = node.execute('TestPipeline', {'source': 'hub', 'value': 'test/model'},
                              'cpu', torch.float32, auto_offload=False, offload_mode='none', **policy)
        return node, output
    return run, loads, instances, entries


def test_normal_native_loader_applies_and_records_policy_then_reuses_only_matching_components(dummy_native_execution):
    run, loads, instances, _ = dummy_native_execution
    first, _ = run(attention_backend='_native_math', vae_slicing=True, vae_tiling=True)
    assert loads == [['transformer', 'vae']]
    assert first._loader_diagnostics['runtime_policy']['applied_controls'] == {
        'attention_backend': '_native_math', 'vae_slicing': True, 'vae_tiling': True,
    }
    assert first._execution_input_record['fields']['attention_backend']['value'] == '_native_math'
    assert first._execution_input_record['fields']['vae_slicing']['value'] is True
    same, _ = run(attention_backend='_native_math', vae_slicing=True, vae_tiling=True)
    assert loads[-1] == []
    assert same.loader.transformer is first.loader.transformer
    assert same.loader.vae is first.loader.vae
    before = list(first.loader.transformer.calls)
    changed, _ = run(attention_backend='auto', vae_slicing=True, vae_tiling=True)
    assert loads[-1] == ['transformer']
    assert changed.loader.transformer is not first.loader.transformer
    assert changed.loader.vae is first.loader.vae
    assert first.loader.transformer.calls == before
    assert len(instances) == 3


def test_normal_native_loader_rejects_connected_policy_conflict_before_pipeline_or_hooks(dummy_native_execution):
    run, loads, instances, entries = dummy_native_execution
    first, output = run(attention_backend='_native_math')
    before = list(first.loader.transformer.calls)
    with pytest.raises(ValueError, match='different attention/VAE policy'):
        run(attention_backend='auto', unet=output['unet_out'])
    assert len(instances) == 1
    assert loads == [['transformer', 'vae']]
    assert first.loader.transformer.calls == before
    assert len(entries) == 2


def test_normal_native_loader_unset_old_policy_does_not_reuse_explicitly_modified_shared_models(dummy_native_execution):
    run, loads, _, _ = dummy_native_execution
    explicit, _ = run(attention_backend='_native_math', vae_slicing=True, vae_tiling=True)
    inherited, _ = run()
    assert loads[-1] == ['transformer', 'vae']
    assert inherited.loader.transformer is not explicit.loader.transformer
    assert inherited.loader.vae is not explicit.loader.vae
    assert inherited.loader.transformer.calls == inherited.loader.vae.calls == []
    assert inherited._loader_diagnostics['runtime_policy']['applied_controls'] == {
        'attention_backend': 'inherit', 'vae_slicing': None, 'vae_tiling': None,
    }


@pytest.mark.parametrize('dummy_native_execution', [True], indirect=True)
def test_normal_native_loader_configures_fresh_vision_and_only_reuses_already_stable_sources(dummy_native_execution, monkeypatch):
    monkeypatch.setattr(torch.version, 'hip', 'test-rocm')
    run, loads, _, _ = dummy_native_execution
    first, _ = run()
    encoder = first.loader.text_encoder
    assert encoder.config.vision_config._attn_implementation == 'eager'
    assert encoder.config.text_config._attn_implementation == 'sdpa'
    assert encoder.vision_calls == [{'vision_config': 'eager'}]
    assert first._loader_diagnostics['runtime_policy']['rocmVisionAttention'] == ['text_encoder']
    reused, _ = run()
    assert reused.loader.text_encoder is encoder and loads[-1] == []
    assert encoder.vision_calls == [{'vision_config': 'eager'}]
    assert reused._loader_diagnostics['runtime_policy']['rocmVisionAttention'] == []
    # A historical/source-owned default SDPA component must reload, rather than
    # be changed in place by a new downstream owner with the same weight ID.
    encoder.config.vision_config._attn_implementation = 'sdpa'
    changed, _ = run()
    assert loads[-1] == ['text_encoder']
    assert changed.loader.text_encoder is not encoder
    assert changed.loader.text_encoder.config.vision_config._attn_implementation == 'eager'
    assert encoder.config.vision_config._attn_implementation == 'sdpa'
    assert encoder.vision_calls == [{'vision_config': 'eager'}]


@pytest.mark.parametrize('dummy_native_execution', [True], indirect=True)
@pytest.mark.parametrize('hip,implementation', [(None, 'sdpa'), ('test-rocm', 'eager'), ('test-rocm', 'flash_attention_2')])
def test_normal_native_loader_preserves_non_rocm_and_alternate_shared_vision_choices(dummy_native_execution, monkeypatch, hip, implementation):
    monkeypatch.setattr(torch.version, 'hip', hip)
    run, loads, _, _ = dummy_native_execution
    first, _ = run()
    encoder = first.loader.text_encoder
    encoder.config.vision_config._attn_implementation = implementation
    before = list(encoder.vision_calls)
    reused, _ = run()
    assert reused.loader.text_encoder is encoder and loads[-1] == []
    assert encoder.config.vision_config._attn_implementation == implementation
    assert encoder.vision_calls == before


def test_unowned_vision_is_not_configured_and_explicit_unstable_source_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.version, 'hip', 'test-rocm')
    encoder = VisionPolicyModule()
    owner = SimpleNamespace(components={'text_encoder': encoder})
    result = loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy())
    assert result['rocmVisionAttention'] == []
    assert encoder.vision_calls == []
    with pytest.raises(ValueError, match='source.*owner'):
        loaders.assert_explicit_component_runtime_policy('text_encoder', encoder, loaders.modular_runtime_policy())
    assert encoder.config.vision_config._attn_implementation == 'sdpa' and encoder.vision_calls == []


def test_fresh_vision_cannot_mutate_an_unowned_alias(monkeypatch):
    monkeypatch.setattr(torch.version, 'hip', 'test-rocm')
    encoder = VisionPolicyModule()
    owner = SimpleNamespace(components={'text_encoder': encoder, 'provided_alias': encoder})
    with pytest.raises(ValueError, match='source.*owner'):
        loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy(), owned_component_names=['text_encoder'])
    assert encoder.config.vision_config._attn_implementation == 'sdpa' and encoder.vision_calls == []
    assert not hasattr(encoder, '_modiff_modular_runtime_policy')


def test_failed_fresh_vision_configuration_invalidates_only_owned_candidates(monkeypatch):
    monkeypatch.setattr(torch.version, 'hip', 'test-rocm')
    encoder, provided = VisionPolicyModule(), VisionPolicyModule()
    transformer = PolicyModule()
    owner = SimpleNamespace(transformer=transformer, components={'text_encoder': encoder, 'provided_encoder': provided,
                                                               'transformer': transformer})
    def fail(value):
        encoder.config.vision_config._attn_implementation = 'eager'
        raise RuntimeError('actual vision setter failed')
    encoder.set_attn_implementation = fail
    with pytest.raises(RuntimeError, match='actual vision setter failed'):
        loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy(), owned_component_names=['text_encoder'])
    assert encoder._modiff_modular_runtime_policy == {'configuration_failed': True}
    assert not compatible(encoder)
    assert not hasattr(provided, '_modiff_modular_runtime_policy')
    assert provided.config.vision_config._attn_implementation == 'sdpa' and provided.vision_calls == []
    assert not hasattr(transformer, '_modiff_modular_runtime_policy')
    assert compatible(transformer)


def test_failed_fresh_vision_preserves_connected_vae_before_any_vae_hook(monkeypatch):
    monkeypatch.setattr(torch.version, 'hip', 'test-rocm')
    encoder, provided_vae = VisionPolicyModule(), PolicyModule()
    provided_vae._modiff_modular_runtime_policy = {'vae_slicing': True}
    owner = SimpleNamespace(vae=provided_vae, components={'text_encoder': encoder, 'vae': provided_vae})
    policy = loaders.modular_runtime_policy(vae_slicing=True)
    loaders.assert_explicit_component_runtime_policy('vae', provided_vae, policy)
    def fail(_value):
        raise RuntimeError('vision rejected before VAE hook')
    encoder.set_attn_implementation = fail
    with pytest.raises(RuntimeError, match='vision rejected before VAE hook'):
        loaders.apply_modular_runtime_policy(owner, policy, owned_component_names=['text_encoder'])
    assert encoder._modiff_modular_runtime_policy == {'configuration_failed': True}
    assert provided_vae._modiff_modular_runtime_policy == {'vae_slicing': True}
    assert provided_vae.calls == []
    assert compatible(provided_vae, {'vae_slicing': True})


def test_real_tiny_transformers_vision_default_is_stabilized_only_by_its_fresh_owner(monkeypatch):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    with torch.random.fork_rng():
        encoder = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(
            text_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1,
                         'num_attention_heads': 2, 'num_key_value_heads': 2, 'vocab_size': 64},
            vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32,
                           'out_hidden_size': 16, 'num_heads': 2, 'patch_size': 2,
                           'spatial_merge_size': 2, 'temporal_patch_size': 1,
                           'fullatt_block_indexes': [0], 'window_size': 4},
        )).eval()
    encoder.set_attn_implementation({'vision_config': 'sdpa'})
    monkeypatch.setattr(torch.version, 'hip', 'test-rocm')
    owner = SimpleNamespace(components={'text_encoder': encoder})
    assert not compatible(encoder)
    loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy())
    assert encoder.config.vision_config._attn_implementation == 'sdpa'
    applied = loaders.apply_modular_runtime_policy(owner, loaders.modular_runtime_policy(), owned_component_names=['text_encoder'])
    assert applied['rocmVisionAttention'] == ['text_encoder']
    assert encoder.config.vision_config._attn_implementation == 'eager'
    assert encoder.config.text_config._attn_implementation == 'sdpa'
    loaders.record_pipeline_component_runtime_policy(owner, offload_mode='none', device='cpu', runtime_policy=loaders.modular_runtime_policy())
    assert compatible(encoder)


@pytest.fixture
def standalone_vision_execution(monkeypatch):
    """Run the actual standalone owner/reuse/publication path without weights."""
    from huggingface_hub import constants as hub_constants

    monkeypatch.setattr(hub_constants, 'HF_HUB_OFFLINE', True)
    entries, loaded, placed, publications, outputs = {}, [], [], [], []
    factory = [VisionPolicyModule]
    identity = ('hub', 'test/vision-component', 'a' * 40, 'transformer',
                'VisionPolicyModule', '1' * 64)

    def load(**kwargs):
        assert kwargs == {'torch_dtype': torch.float32, 'local_files_only': True}
        model = factory[0]()
        loaded.append(model)
        return model

    def add(name, model, collection):
        publications.append(model)
        key = next((key for key, current in entries.items() if current is model), None)
        if key is None:
            key = f'standalone-vision:{id(entries)}:{len(entries)}'
            entries[key] = model
        return key

    manager = SimpleNamespace(
        _lookup_ids=lambda **_: list(entries),
        get_one=lambda component_id: entries[component_id],
        add=add,
        get_model_info=lambda key: {'model_id': key, 'class_name': identity[4]},
        remove_from_collection=lambda *_: None,
    )

    def place(model, **kwargs):
        placed.append((model, model.config.vision_config._attn_implementation))
        return SimpleNamespace(mode=kwargs['mode'], method='tiny-cpu-fixture',
                               components=[kwargs['component_name']])

    monkeypatch.setattr(loaders, 'components', manager)
    monkeypatch.setattr(loaders, '_preflight_reviewed_diffusers_component', lambda *_: identity)
    monkeypatch.setattr(loaders, '_resolve_reviewed_diffusers_component_class', lambda *_: VisionPolicyModule)
    monkeypatch.setattr(loaders, 'ComponentSpec', lambda **_: SimpleNamespace(load_id='exact-vision-weight-id', load=load))
    monkeypatch.setattr(loaders, 'apply_model_offload', place)

    def run():
        node = loaders.AutoModelLoader(f'standalone-vision-owner:{len(outputs)}')
        monkeypatch.setattr(node, 'progress', lambda *_args, **_kwargs: None)
        output = node.execute('transformer', {'source': 'hub', 'value': identity[1]},
                              torch.float32, False, device='cpu', auto_offload=False,
                              offload_mode='none', subfolder='transformer', revision=identity[2])
        # Retain the real sealed publication so the next owner exercises the
        # actual manager provenance check, rather than a mocked reuse result.
        outputs.append((node, output))
        return output

    return run, factory, loaded, placed, publications, entries


def test_standalone_source_owner_configures_real_tiny_vision_before_placement_and_reuses_it(standalone_vision_execution, monkeypatch):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    run, factory, loaded, placed, publications, _ = standalone_vision_execution
    def tiny_encoder():
        with torch.random.fork_rng():
            model = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(
                text_config={'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1,
                             'num_attention_heads': 2, 'num_key_value_heads': 2, 'vocab_size': 64},
                vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32,
                               'out_hidden_size': 16, 'num_heads': 2, 'patch_size': 2,
                               'spatial_merge_size': 2, 'temporal_patch_size': 1,
                               'fullatt_block_indexes': [0], 'window_size': 4},
            )).eval()
        model.set_attn_implementation({'vision_config': 'sdpa'})
        return model
    factory[0] = tiny_encoder
    monkeypatch.setattr(torch.version, 'hip', 'test-rocm')
    first = run()
    encoder = loaded[0]
    assert placed == [(encoder, 'eager')]
    assert encoder.config.text_config._attn_implementation == 'sdpa'
    second = run()
    assert first['model']['model_id'] == second['model']['model_id']
    assert len(loaded) == len(placed) == 1 and publications == [encoder, encoder]
    # An upstream-owned module that became unstable must be reloaded, never
    # reconfigured by the downstream owner that requested these same weights.
    encoder.set_attn_implementation({'vision_config': 'sdpa'})
    third = run()
    assert third['model']['model_id'] != second['model']['model_id']
    assert len(loaded) == len(placed) == 2
    assert placed[-1] == (loaded[-1], 'eager')
    assert encoder.config.vision_config._attn_implementation == 'sdpa'


@pytest.mark.parametrize('hip,implementation', [(None, 'sdpa'), ('test-rocm', 'eager'), ('test-rocm', 'flash_attention_2')])
def test_standalone_owner_preserves_non_rocm_and_alternate_vision_choices(standalone_vision_execution, monkeypatch, hip, implementation):
    run, factory, loaded, placed, _, _ = standalone_vision_execution
    def selected_encoder():
        encoder = VisionPolicyModule()
        encoder.config.vision_config._attn_implementation = implementation
        return encoder
    factory[0] = selected_encoder
    monkeypatch.setattr(torch.version, 'hip', hip)
    first, second = run(), run()
    assert first['model']['model_id'] == second['model']['model_id']
    assert len(loaded) == len(placed) == 1
    assert placed[0] == (loaded[0], implementation) and loaded[0].vision_calls == []


def test_standalone_fresh_vision_setter_failure_never_places_or_publishes(standalone_vision_execution, monkeypatch):
    run, factory, loaded, placed, publications, entries = standalone_vision_execution
    monkeypatch.setattr(torch.version, 'hip', 'test-rocm')
    run()
    source = loaded[0]
    source_calls = list(source.vision_calls)
    current_entries = dict(entries)
    # Force another fresh owner load while retaining the existing publication.
    source.config.vision_config._attn_implementation = 'sdpa'
    def failing_encoder():
        encoder = VisionPolicyModule()
        def fail(_value):
            encoder.config.vision_config._attn_implementation = 'eager'
            raise RuntimeError('fresh standalone vision setter failed')
        encoder.set_attn_implementation = fail
        return encoder
    factory[0] = failing_encoder
    with pytest.raises(RuntimeError, match='fresh standalone vision setter failed'):
        run()
    assert len(loaded) == 2 and len(placed) == len(publications) == 1
    assert entries == current_entries
    assert loaded[-1]._modiff_modular_runtime_policy == {'configuration_failed': True}
    assert not compatible(loaded[-1])
    assert source.config.vision_config._attn_implementation == 'sdpa' and source.vision_calls == source_calls
