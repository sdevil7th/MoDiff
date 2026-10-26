"""Exact FLUX 3 source-audit contracts; no pretrained artifacts are needed."""
import pytest

from modules.ModularDiffusers import modular_utils


@pytest.fixture
def flux3_api():
    import diffusers
    for name in ('Flux3ModularPipeline', 'Flux3DistilledModularPipeline', 'Flux3Transformer2DModel'):
        if getattr(diffusers, name, None) is None:
            pytest.skip('FLUX 3 source APIs are absent from the reviewed installed release; run the pinned source audit.')


@pytest.mark.parametrize('name,steps,negative,guider', [
    ('Flux3ModularPipeline', 50, True, True),
    ('Flux3DistilledModularPipeline', 16, False, False),
])
def test_registered_schema_matches_actual_native_blocks(name, steps, negative, guider, flux3_api):
    import diffusers
    pipeline_class = getattr(diffusers, name)
    metadata = modular_utils.get_model_type_metadata(name)
    assert metadata is not None, f'{name} has no backend native schema'
    pipeline = pipeline_class()
    assert set(pipeline.blocks.available_workflows) == {'text2image', 'image_conditioned'}
    text = metadata['node_params']['text_encoder']
    denoise = metadata['node_params']['denoise']
    encoder = metadata['node_params']['vae_encoder']
    decoder = metadata['node_params']['decoder']
    assert ('negative_prompt' in text['input_names']) is negative
    assert ('guider' in denoise['model_input_names']) is guider
    assert 'guider' not in text['model_input_names']
    assert 'guidance_scale' not in denoise['input_names']
    assert denoise['params']['num_inference_steps']['default'] == steps
    assert text['params']['max_sequence_length']['default'] == 8192
    assert tuple(text['params']['text_encoder_out_layers']['default']) == (4, 8, 12, 16, 20, 24, 28, 32)
    assert text['params']['text_encoder_out_layers']['display'] == 'select'
    assert 'seed' in denoise['input_names'] and 'seed' not in encoder['input_names']
    assert encoder['output_names'] == ['image_cond_latents', 'reference_binding', 'doc']
    assert 'image_cond_latents' in denoise['input_names']
    assert 'image_latents' not in denoise['input_names']
    assert decoder['input_names'] == ['latents']
    for action in ('text_encoder', 'vae_encoder', 'denoise', 'decoder'):
        blocks, config = modular_utils.require_modiff_node_contract(pipeline_class, action)
        assert blocks is not None
        native_inputs = set(blocks.input_names)
        for field in config['input_names']:
            assert field in native_inputs or field in {'seed', 'embeddings'}, (name, action, field)
    if negative:
        assert text['params']['negative_prompt']['default'] == ''
        assert pipeline.guider.guidance_scale == 3.5
    else:
        assert 'guider' not in pipeline.component_names


def _layer_schema():
    # Pure declaration checks also run before the future APIs are installed.
    return modular_utils.FLUX_3_PIPELINE_CONFIG.node_params['text_encoder']


def test_declared_layer_tuple_normalizes_native_values_and_preserves_order():
    schema = _layer_schema()
    authored = ['32', '28', '24', '20', '16', '12', '8', '4']
    actual = modular_utils.normalize_modular_runtime_params({'text_encoder_out_layers': authored}, schema)
    assert actual['text_encoder_out_layers'] == (32, 28, 24, 20, 16, 12, 8, 4)
    assert authored == ['32', '28', '24', '20', '16', '12', '8', '4']


@pytest.mark.parametrize('value', [
    [True, 8, 12, 16, 20, 24, 28, 32],
    [4.0, 8, 12, 16, 20, 24, 28, 32],
    ['04', 8, 12, 16, 20, 24, 28, 32],
    [' 4', 8, 12, 16, 20, 24, 28, 32],
    [4, 4, 12, 16, 20, 24, 28, 32],
    [4, 8, 12, 16, 20, 24, 28, 33],
    [4, 8], '4,8,12,16,20,24,28,32', None,
])
def test_invalid_layer_sequences_fail_before_the_official_call(value):
    schema = _layer_schema()
    if value is None:
        # An absent optional value preserves the exact official default.
        assert modular_utils.normalize_modular_runtime_params({'text_encoder_out_layers': value}, schema)[
            'text_encoder_out_layers'] is None
    else:
        with pytest.raises(ValueError, match='integer sequence'):
            modular_utils.normalize_modular_runtime_params({'text_encoder_out_layers': value}, schema)


def test_real_encoder_hidden_layer_geometry_matches_selected_tuple():
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    schema = _layer_schema()
    selected = tuple(schema['params']['text_encoder_out_layers']['default'])
    config = Qwen3VLConfig(
        text_config={'vocab_size': 16, 'hidden_size': 8, 'intermediate_size': 16,
                     'num_hidden_layers': 32, 'num_attention_heads': 1,
                     'num_key_value_heads': 1, 'head_dim': 8,
                     'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000,
                                         'mrope_section': [1, 1, 2], 'mrope_interleaved': True}},
        vision_config={'depth': 1, 'hidden_size': 8, 'intermediate_size': 16,
                       'num_heads': 1, 'out_hidden_size': 8, 'patch_size': 2,
                       'spatial_merge_size': 1, 'temporal_patch_size': 1},
        image_token_id=12, video_token_id=13, vision_start_token_id=14, vision_end_token_id=15,
    )
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(9)
        encoder = Qwen3VLForConditionalGeneration(config).eval()
        ids = torch.tensor([[1, 2, 3]])
        output = encoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                         position_ids=torch.arange(3)[None], output_hidden_states=True,
                         use_cache=False)
        assert len(output.hidden_states) == config.text_config.num_hidden_layers + 1
        stacked = torch.stack([output.hidden_states[index] for index in selected], dim=1)
        assert stacked.shape == (1, 8, 3, 8)
        assert torch.isfinite(stacked).all()


def _encoding_observation(monkeypatch, *, declared=True):
    from types import SimpleNamespace
    from modules.ModularDiffusers.embeddings import EncodePrompt
    calls = []

    class Pipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return {}

    pipeline = Pipeline()
    blocks = SimpleNamespace(input_names=['prompt'], component_names=[],
                             init_pipeline=lambda **kwargs: pipeline)
    params = {'prompt': {'type': 'string'}}
    if declared:
        params['layout_conditioning'] = {
            'type': 'layout_conditioning', 'display': 'layout', 'default': None,
            'fieldOptions': {'layoutConditioning': {'schemaVersion': 1, 'sourceRegions': False}},
        }
    schema = {'params': params, 'input_names': ['prompt'], 'model_input_names': ['text_encoders'],
              'output_names': []}
    monkeypatch.setattr('modules.ModularDiffusers.embeddings.require_modiff_node_contract',
                        lambda *args, **kwargs: (blocks, schema))
    monkeypatch.setattr('modules.ModularDiffusers.embeddings.collect_model_ids', lambda *args, **kwargs: [])
    node = EncodePrompt()
    return node, calls


def _layout(source=False):
    region = {'id': 'r1', 'description': 'red cup', 'targetBox': [0.1, 0.2, 0.4, 0.7]}
    if source:
        region['source'] = {'referenceIndex': 0, 'box': [0.0, 0.0, 1.0, 1.0]}
    return {'schemaVersion': 1, 'coordinateSpace': 'normalized_xyxy', 'regions': [region]}


def test_declared_layout_projects_actual_consumed_connected_prompt(monkeypatch, flux3_api):
    from modiff.layout_conditioning import resolve_layout_prompt
    node, calls = _encoding_observation(monkeypatch)
    authored = _layout()
    node(prompt='unchanged saved inline', prompt_input='A scene with <r1>', layout_conditioning=authored,
         text_encoders={'model_type': 'Flux3ModularPipeline', 'repo_id': 'source-audit/no-artifact'})
    expected = resolve_layout_prompt('A scene with <r1>', authored)
    assert calls == [{'prompt': expected}]
    assert node._execution_input_record['fields']['prompt']['value'] == expected
    assert node._execution_input_source_fields == {'prompt': 'prompt_input'}
    assert authored == _layout()


def test_layout_none_preserves_raw_prompt_bytes(monkeypatch, flux3_api):
    node, calls = _encoding_observation(monkeypatch)
    raw = '  raw prompt\n<tag> {unparsed}\n'
    node(prompt=raw, layout_conditioning=None,
         text_encoders={'model_type': 'Flux3ModularPipeline', 'repo_id': 'source-audit/no-artifact'})
    assert calls == [{'prompt': raw}]


def test_undeclared_layout_refuses_before_any_pipeline_call(monkeypatch, flux3_api):
    node, calls = _encoding_observation(monkeypatch, declared=False)
    with pytest.raises(ValueError, match='layout'):
        node(prompt='A scene with <r1>', layout_conditioning=_layout(),
             text_encoders={'model_type': 'Flux3ModularPipeline', 'repo_id': 'source-audit/no-artifact'})
    assert calls == []


@pytest.mark.parametrize('pipeline', ['Flux3ModularPipeline', 'Flux3DistilledModularPipeline'])
def test_selected_component_precision_map_is_declared_and_legacy_scalar_stays_identical(pipeline, flux3_api):
    import torch
    from modules.ModularDiffusers import loaders
    resolver = getattr(loaders, 'reviewed_component_dtypes', None)
    assert callable(resolver), 'Selected component precision is not connected to the native loader'
    result = resolver(pipeline, torch.bfloat16)
    assert result == {'default': torch.bfloat16, 'vae': torch.float32}
    assert resolver('Flux2KleinModularPipeline', torch.bfloat16) is torch.bfloat16
    assert resolver('unregistered', torch.float16) is torch.float16


@pytest.mark.parametrize('value', [None, {}, [['vae', 'float8']], [['vae', True]],
                                  [['vae', 'float32'], ['vae', 'bfloat16']], [['__proto__', 'float32']],
                                  [['vae']], [['vae', 'float32', 'extra']]])
def test_component_precision_metadata_refuses_malformed_or_unsupported_maps(value):
    from modules.ModularDiffusers import pipeline_schema
    validator = getattr(pipeline_schema, 'normalize_component_dtype_overrides', None)
    assert callable(validator), 'Component precision metadata has no strict validator'
    with pytest.raises(ValueError, match='dtype'):
        validator(value)


def test_component_precision_schema_round_trip_preserves_old_projection():
    from modules.ModularDiffusers.pipeline_schema import MoDiffPipelineConfig
    old = modular_utils.FLUX_2_KLEIN_DISTILLED_PIPELINE_CONFIG.to_dict()
    assert 'component_dtype_overrides' not in old
    assert MoDiffPipelineConfig.from_dict(old).to_dict() == old
    candidate = modular_utils.FLUX_3_PIPELINE_CONFIG.to_dict()
    assert candidate.get('component_dtype_overrides') == [['vae', 'float32']]
    assert MoDiffPipelineConfig.from_dict(candidate).to_dict() == candidate


def test_official_component_loading_honors_mixed_precision(tmp_path, flux3_api):
    import torch
    from diffusers import AutoencoderKLFlux2, Flux3DistilledModularPipeline, Flux3Transformer2DModel
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
    from modules.ModularDiffusers import loaders
    resolver = getattr(loaders, 'reviewed_component_dtypes', None)
    assert callable(resolver), 'Selected component precision is not connected to the native loader'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        vae = AutoencoderKLFlux2(
            block_out_channels=(4,), down_block_types=('DownEncoderBlock2D',),
            up_block_types=('UpDecoderBlock2D',), layers_per_block=1,
            latent_channels=2, norm_num_groups=1, mid_block_add_attention=False,
        )
        transformer = Flux3Transformer2DModel(
            in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=1, joint_attention_dim=64,
            timestep_channels=8, axes_dims_rope=(4, 4, 4, 4),
        )
    vae.save_pretrained(tmp_path / 'vae', safe_serialization=True)
    transformer.save_pretrained(tmp_path / 'transformer', safe_serialization=True)
    pipeline = Flux3DistilledModularPipeline()
    pipeline._component_specs['vae'] = ComponentSpec('vae', AutoencoderKLFlux2,
                                                    pretrained_model_name_or_path=str(tmp_path), subfolder='vae')
    pipeline._component_specs['transformer'] = ComponentSpec(
        'transformer', Flux3Transformer2DModel, pretrained_model_name_or_path=str(tmp_path), subfolder='transformer')
    policy = resolver('Flux3DistilledModularPipeline', torch.bfloat16)
    diagnostics = {}
    loaders.load_components_strict(
        pipeline, ['vae', 'transformer'], required_names={'vae', 'transformer'},
        model_id='owned-local-tiny-precision-fixture', dtype=torch.bfloat16,
        offload_mode='none', quant_config=None, diagnostics=diagnostics,
        component_load_kwargs={'torch_dtype': policy, 'local_files_only': True},
    )
    assert pipeline.vae.dtype == torch.float32
    assert pipeline.transformer.dtype == torch.bfloat16
    assert set(diagnostics['components_loaded']) == {'vae', 'transformer'}
    # The actual official loaded tensors retain the serialized values within
    # the explicitly requested BF16 conversion tolerance.
    torch.testing.assert_close(next(pipeline.vae.parameters()), next(vae.parameters()), rtol=0, atol=0)
    torch.testing.assert_close(next(pipeline.transformer.parameters()).float(),
                               next(transformer.parameters()), rtol=0.01, atol=0.001)


def test_layout_source_region_has_no_invented_reference_authority(monkeypatch, flux3_api):
    node, calls = _encoding_observation(monkeypatch)
    with pytest.raises(ValueError, match='reference'):
        node(prompt='A scene with <r1>', layout_conditioning=_layout(source=True),
             text_encoders={'model_type': 'Flux3ModularPipeline', 'repo_id': 'source-audit/no-artifact'})
    assert calls == []


def test_retained_private_specs_use_actual_registered_ports_and_stay_unpublished(monkeypatch, flux3_api):
    import modules
    from modiff import studio_execution_specs as specs
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    private_ids = {f'{variant}:modular:{mode}:v1'
                   for variant in ('flux3-klein', 'flux3-klein-base')
                   for mode in ('text-to-image', 'edit-image')}
    selected = {key: value for key, value in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.items()
                if value['profile']['id'] in {'flux3-klein:modular', 'flux3-klein-base:modular'}}
    assert set(selected) == private_ids
    monkeypatch.setattr(specs, 'STUDIO_EXECUTION_SPEC_DEFINITIONS', selected)
    validated = specs.validate_studio_execution_specs(modules.MODULE_MAP, include_private=True)
    assert len(validated) == 4
    for definition in selected.values():
        profile = DIFFUSERS_EXECUTION_PROFILES[definition['profile']['id']]
        assert profile.public is False
        assert profile.live_proof is False
        assert profile.quantizable_components == ()
        assert profile.adapter_authoring is None
        capability = definition['capability']
        for key in ('autoEligible', 'templateEligible', 'galleryEligible', 'liveProof', 'supportsLora'):
            assert capability[key] is False
        assert capability['executionStatus'] == 'contract_only'
        assert capability['qualifiedModes'] == []
        if definition['modelType'] == 'Flux3ModularPipeline':
            assert ('guider', 'guider_out', 'denoise', 'guider') in definition['edges']
            assert ('guider', 'guider', 'classifierFreeGuidance') in definition['bindings']
            assert ('guider', 'model_type', 'pipelineClass') in definition['bindings']


def test_required_component_error_reports_actual_component_dtype(tmp_path, flux3_api):
    import torch
    from diffusers import AutoencoderKLFlux2, Flux3DistilledModularPipeline
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
    from modules.ModularDiffusers import loaders
    vae = AutoencoderKLFlux2(
        block_out_channels=(4,), down_block_types=('DownEncoderBlock2D',),
        up_block_types=('UpDecoderBlock2D',), layers_per_block=1,
        latent_channels=2, norm_num_groups=1, mid_block_add_attention=False,
    )
    vae.save_pretrained(tmp_path / 'vae', safe_serialization=True)
    weights = tmp_path / 'vae/diffusion_pytorch_model.safetensors'
    assert weights.is_file()
    weights.write_bytes(b'owned deliberately invalid safetensors fixture')
    pipeline = Flux3DistilledModularPipeline()
    pipeline._component_specs['vae'] = ComponentSpec(
        'vae', AutoencoderKLFlux2, pretrained_model_name_or_path=str(tmp_path), subfolder='vae')
    diagnostics = {}
    with pytest.raises(loaders.RequiredComponentLoadError) as caught:
        loaders.load_components_strict(
            pipeline, ['vae'], required_names={'vae'}, model_id='owned-corrupt-local-fixture',
            dtype=torch.bfloat16, offload_mode='none', quant_config=None, diagnostics=diagnostics,
            component_load_kwargs={'torch_dtype': {'default': torch.bfloat16, 'vae': torch.float32},
                                   'local_files_only': True},
        )
    assert caught.value.component_name == 'vae'
    assert caught.value.dtype == str(torch.float32)
    assert diagnostics['components_failed'][0]['load_kwargs']['torch_dtype'] == str(torch.float32)


def test_component_reuse_respects_selected_precision_without_mutating_tensors(flux3_api):
    import torch
    from diffusers import ComponentsManager
    from modules.ModularDiffusers import loaders
    component = torch.nn.Linear(2, 2, bias=False).float()
    manager = ComponentsManager()
    component._diffusers_load_id = 'owned-tiny-precision'
    manager.add('vae', component)
    common = {'manager': manager, 'name': 'vae', 'load_id': 'owned-tiny-precision',
              'requested_quantization': None, 'offload_mode': 'none', 'device': 'cpu'}
    policy = loaders.reviewed_component_dtypes('Flux3ModularPipeline', torch.bfloat16)
    assert loaders.reusable_component_ids(dtype=policy, **common)
    assert loaders.reusable_component_ids(dtype=torch.bfloat16, **common) == []
    component.to(dtype=torch.bfloat16)
    assert loaders.reusable_component_ids(dtype=policy, **common) == []
    assert loaders.reusable_component_ids(dtype=torch.bfloat16, **common)


def test_current_public_four_specs_validate_registered_ports_without_optional_runtime(monkeypatch):
    """Current discoverable declarations use real ports; authoring grants no fit/output proof."""
    import modules
    from modiff import studio_execution_specs as specs
    from modiff.NodeBase import NodeBase
    from modiff.optional_runtimes import DIFFUSERS_FLUX3_RUNTIME_PROFILE_ID
    from modules.ModularDiffusers import modular_utils

    public_profiles = {'flux3-klein:official-provider:v1', 'flux3-klein-base:official-provider:v1'}
    private_profiles = {'flux3-klein:modular', 'flux3-klein-base:modular'}
    expected = {f'{profile}:{mode}:v1' for profile in public_profiles
                for mode in ('text-to-image', 'edit-image')}
    # Keep exact legacy definitions for the maintained replacement validator;
    # ordinary validation must select only the four current public definitions.
    selected = {key: value for key, value in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.items()
                if value['profile']['id'] in public_profiles | private_profiles}
    assert len(selected) == 8
    monkeypatch.setattr(specs, 'STUDIO_EXECUTION_SPEC_DEFINITIONS', selected)
    def forbidden(*_args, **_kwargs):
        pytest.fail('Public spec validation attempted node construction or installed pipeline resolution')
    monkeypatch.setattr(NodeBase, '__init__', forbidden)
    monkeypatch.setattr(modular_utils, 'pipeline_class_from_model_type', forbidden)
    validated = specs.validate_studio_execution_specs(modules.MODULE_MAP)
    assert {row['id'] for row in validated} == expected
    assert len(validated) == 4
    for row in validated:
        definition = selected[row['id']]
        profile, capability = definition['profile'], definition['capability']
        assert profile['public'] is True and profile['live_proof'] is False
        assert row['executionProfileId'] in public_profiles
        assert row['requiredDiffusersProvider'] == DIFFUSERS_FLUX3_RUNTIME_PROFILE_ID
        assert capability['executionStatus'] == 'expert_only'
        assert capability['qualificationStatus'] == 'runtime-required'
        assert all(capability[key] is False for key in (
            'autoEligible', 'templateEligible', 'galleryEligible', 'liveProof'))
