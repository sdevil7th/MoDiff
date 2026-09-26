"""Typed attention options preserve upstream ownership and ordinary node reuse."""
import pytest
import torch


def test_builder_is_an_ordinary_node_and_preserves_mask_identity():
    from modules.Tensor.main import AttentionArguments
    from modules.DiffusersImage.call_inputs import normalize_call_inputs
    assert AttentionArguments.params['attention_mask']['display'] == 'input'
    assert AttentionArguments.params['options']['display'] == 'output'
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    node = AttentionArguments('attention-options')
    assert node.execute() == {'options': {}}
    args = node.execute(attention_mask=mask, enable_lora_scale=True, lora_scale=.35)['options']
    assert args['attention_mask'] is mask and args['scale'] == .35
    for pipeline, field in [('FluxPipeline', 'joint_attention_kwargs'), ('Flux2Pipeline', 'attention_kwargs'),
                            ('Flux2KleinPipeline', 'attention_kwargs')]:
        normalized = normalize_call_inputs(pipeline, {field: args})[field]
        assert normalized is not args and normalized['attention_mask'] is mask
        assert normalized['scale'] == .35
    with pytest.raises(ValueError, match='Klein.*KV|KleinKV'):
        normalize_call_inputs('Flux2KleinKVPipeline', {'attention_kwargs': args})
    assert set(args) == {'scale', 'attention_mask'}


@pytest.mark.parametrize('mask', [
    'not a tensor', [[True]], torch.tensor(True), torch.ones(3), torch.ones(1, 1, 1, 1, 1),
    torch.ones(2, 2, dtype=torch.long), torch.empty(0, 2), torch.empty(2, 2, device='meta'),
    torch.ones(2, 2).to_sparse(), torch.full((2, 2), float('nan')), torch.full((2, 2), float('inf')),
])
def test_invalid_mask_names_the_actual_input(mask):
    from modules.Tensor.main import AttentionArguments
    from modules.DiffusersImage.call_inputs import normalize_call_inputs
    with pytest.raises(ValueError, match='attention_mask'):
        AttentionArguments('bad-attention').execute(attention_mask=mask)
    with pytest.raises(ValueError, match='attention_mask'):
        normalize_call_inputs('FluxPipeline', {'joint_attention_kwargs': {'attention_mask': mask}})


def test_additive_negative_infinity_masks_and_absent_scale_are_preserved():
    from modules.Tensor.main import AttentionArguments
    additive = torch.tensor([[0., float('-inf')], [0., 0.]])
    options = AttentionArguments('additive-mask').execute(attention_mask=additive, lora_scale=.1)['options']
    assert options['attention_mask'] is additive
    assert 'scale' not in options
    for value in [True, float('nan'), float('inf'), 101, 'not numeric']:
        with pytest.raises(ValueError, match='scale'):
            AttentionArguments('invalid-scale').execute(enable_lora_scale=True, lora_scale=value)


def test_upstream_ignored_mask_and_mutable_execution_objects_are_not_admitted():
    from modules.DiffusersImage.call_inputs import normalize_call_inputs
    for key in ['ip_adapter_masks', 'kv_cache', 'kv_cache_mode', 'processor', 'callback']:
        with pytest.raises(ValueError, match='joint_attention_kwargs'):
            normalize_call_inputs('FluxPipeline', {'joint_attention_kwargs': {key: object()}})
