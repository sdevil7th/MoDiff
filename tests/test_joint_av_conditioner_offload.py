"""Source-declared joint conditioner entrypoints use actual tiny official models."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from modiff import diffusers_offload as offload
from modules.ModularDiffusers import modular_utils


def test_entrypoint_policy_is_source_declared_and_does_not_change_other_models():
    for pipeline in ('MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline'):
        declared = modular_utils.component_group_offload_types(pipeline)
        assert declared == {'text_encoder': 'leaf_level'}
        declared['text_encoder'] = 'block_level'
        assert modular_utils.component_group_offload_types(pipeline) == {'text_encoder': 'leaf_level'}
    assert modular_utils.component_group_offload_types('FluxModularPipeline') == {}
    assert modular_utils.component_group_offload_types('CustomModularPipeline') == {}


def test_component_reuse_refuses_a_different_source_owned_entrypoint_policy():
    from modules.ModularDiffusers import loaders
    component = torch.nn.Linear(2, 2)
    request = {'component_group_offload_types': {'text_encoder': 'leaf_level'}}
    policy = loaders._component_runtime_policy('text_encoder', request)
    assert policy == {'group_offload_type': 'leaf_level'}
    assert loaders._component_runtime_policy('transformer', request) == {}
    assert loaders._component_runtime_policy('vae', request) == {}
    manager = SimpleNamespace(_lookup_ids=lambda **kwargs: ['borrowed'], get_one=lambda **kwargs: component)
    options = dict(name='text_encoder', load_id='same-weights', dtype=torch.float32,
                   requested_quantization=None, offload_mode='none', device='cpu', runtime_policy=request)
    assert loaders.reusable_component_ids(manager, **options) == []
    component._modiff_modular_runtime_policy = policy
    assert loaders.reusable_component_ids(manager, **options) == ['borrowed']


@pytest.mark.parametrize('pipeline', ['MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline'])
def test_declared_joint_conditioner_nested_forward_retains_exact_weights(tmp_path, monkeypatch, pipeline):
    config = (modular_utils.MINIMAX_H3_PIPELINE_CONFIG if pipeline == 'MiniMaxH3ModularPipeline'
              else modular_utils.PIKA_H3_FLEX_PIPELINE_CONFIG)
    metadata = modular_utils._pipeline_config_metadata(pipeline, config)
    # CPU-to-CPU exercises actual disk hook storage and nested entrypoints.
    # No accelerator or trained-checkpoint qualification follows from this.
    monkeypatch.setattr(offload, 'supports_accelerator_cpu_offload', lambda device: True)
    monkeypatch.chdir(tmp_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(71)
            encoder = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
                text_config={'vocab_size': 16, 'hidden_size': 8, 'intermediate_size': 16,
                    'num_hidden_layers': 2, 'num_attention_heads': 1, 'num_key_value_heads': 1,
                    'head_dim': 8, 'use_cache': False,
                    'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2]}},
                vision_config={'depth': 1, 'hidden_size': 8, 'intermediate_size': 16,
                    'num_heads': 1, 'out_hidden_size': 8, 'patch_size': 2,
                    'spatial_merge_size': 1, 'temporal_patch_size': 1},
                image_token_id=12, video_token_id=13, vision_start_token_id=14, vision_end_token_id=15,
            )).eval()
            ids = torch.tensor([[1, 2, 3]])
            expected = encoder.model(input_ids=ids, use_cache=False).last_hidden_state.clone()
            declared = metadata.get('component_group_offload_types', {})
            leaf = offload.LEAF_LEVEL_GROUP_COMPONENTS | {
                name for name, strategy in declared.items() if strategy == 'leaf_level'}
            result = offload.apply_component_group_offload(SimpleNamespace(text_encoder=encoder),
                component_names=['text_encoder'], device='cpu', mode='group_disk', node_id='tiny-conditioner',
                leaf_level_components=leaf)
            assert result.applied
            actual = encoder.model(input_ids=ids, use_cache=False).last_hidden_state
            torch.testing.assert_close(actual, expected)
    finally:
        torch.set_num_threads(previous_threads)
