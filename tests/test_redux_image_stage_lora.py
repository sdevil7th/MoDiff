"""Real official split transformer/CLIP LoRA mutation and refusal contracts."""
from copy import deepcopy
import hashlib

import pytest

from redux_stage_test_support import actual_tiny_bundle as _actual_tiny_bundle
actual_tiny_bundle = _actual_tiny_bundle
from redux_stage_test_support import bounded_redux_stage_runtime as bounded_redux_stage_runtime


def serialize_actual_joint_lora(tmp_path, bundle, name):
    from diffusers import FluxPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    transformer = deepcopy(bundle.base.transformer)
    clip = deepcopy(bundle.prior.text_encoder)
    transformer.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=['to_q']))
    clip.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=['q_proj', 'v_proj']))
    for model in (transformer, clip):
        for parameter_name, parameter in model.named_parameters():
            if "lora_B" in parameter_name:
                parameter.data.fill_(.125)
    transformer_state = get_peft_model_state_dict(transformer)
    text_state = get_peft_model_state_dict(clip)
    assert transformer_state and text_state
    path = tmp_path / (name + '.safetensors')
    FluxPipeline.save_lora_weights(tmp_path, weight_name=path.name,
        transformer_lora_layers=transformer_state, text_encoder_lora_layers=text_state,
        safe_serialization=True)
    assert 0 < path.stat().st_size < 1024**2
    return path


def actual_load(owner, path, name, scale, *, replace=False):
    from modules.DiffusersImage.main import LoadAdapter
    return LoadAdapter('joint-actual-' + name)(pipeline=owner,
        adapter_path={'source': 'local', 'value': str(path)}, weight_name=path.name,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        adapter_name=name, scale=scale, replace_existing=replace)['output']


@pytest.mark.parametrize('kind', ['official_unified_base_control', 'actual_redux_bundle'])
def test_joint_serialized_adapter_reaches_both_real_owners_without_text_duplication(
    actual_tiny_bundle, tmp_path, kind,
):
    bundle = actual_tiny_bundle
    if kind == 'official_unified_base_control':
        # An independent real official control pipeline owns its own copied CLIP.
        # This is a test control, never the proposed production bridge.
        owner = deepcopy(bundle.base)
        owner.register_modules(text_encoder=deepcopy(bundle.prior.text_encoder))
        clip = owner.text_encoder
        transformer = owner.transformer
    else:
        owner = bundle
        clip, transformer = bundle.prior.text_encoder, bundle.base.transformer
        assert bundle.base.text_encoder is None
    t5 = bundle.prior.text_encoder_2
    path = serialize_actual_joint_lora(tmp_path, bundle, 'joint')
    assert actual_load(owner, path, 'first', .35) is owner
    assert 'first' in transformer.peft_config and 'first' in clip.peft_config
    assert actual_load(owner, path, 'second', .55) is owner
    assert owner._modiff_adapter_scales == {'first': .35, 'second': .55}
    assert {'first', 'second'} <= set(transformer.peft_config)
    assert {'first', 'second'} <= set(clip.peft_config)
    assert not getattr(t5, 'peft_config', {})
    if kind == 'actual_redux_bundle':
        assert bundle.base.text_encoder is None
        assert bundle.prior.text_encoder is clip
        assert bundle.prior.text_encoder_2 is t5
    assert actual_load(owner, path, 'replacement', .25, replace=True) is owner
    assert owner._modiff_adapter_scales == {'replacement': .25}
    assert set(transformer.peft_config) == {'replacement'}
    assert set(clip.peft_config) == {'replacement'}
    if kind == 'actual_redux_bundle':
        assert bundle.base.text_encoder is None and bundle.prior.text_encoder is clip


@pytest.mark.parametrize('prefix', ['text_encoder_2.', 'foreign_component.'])
def test_redux_bridge_rejects_unknown_joint_inventory_before_any_official_mutation(
    actual_tiny_bundle, tmp_path, prefix,
):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    bundle = actual_tiny_bundle
    path = serialize_actual_joint_lora(tmp_path, bundle, 'valid')
    with safe_open(path, framework='pt', device='cpu') as reader:
        state = {key: reader.get_tensor(key) for key in reader.keys()}
    state[prefix + 'unreviewed.lora_A.weight'] = torch.zeros(2, 8)
    invalid = tmp_path / 'invalid.safetensors'
    save_file(state, invalid)
    before_transformer = {key: value.detach().clone() for key, value in bundle.base.transformer.state_dict().items()}
    before_clip = {key: value.detach().clone() for key, value in bundle.prior.text_encoder.state_dict().items()}
    with pytest.raises((ValueError, RuntimeError), match='adapter|prefix|component|LoRA'):
        actual_load(bundle, invalid, 'bad', .35)
    assert not getattr(bundle.base.transformer, 'peft_config', {})
    assert not getattr(bundle.prior.text_encoder, 'peft_config', {})
    assert getattr(bundle, '_modiff_adapter_scales', {}) == {}
    for key, value in before_transformer.items():
        torch.testing.assert_close(bundle.base.transformer.state_dict()[key], value, rtol=0, atol=0)
    for key, value in before_clip.items():
        torch.testing.assert_close(bundle.prior.text_encoder.state_dict()[key], value, rtol=0, atol=0)


@pytest.fixture(autouse=True)
def selected_actual_redux_owner(actual_tiny_bundle):
    from modules.DiffusersImage import main, stage_execution as core
    bundle = actual_tiny_bundle
    main._tag_image_pipeline(bundle, main.IMAGE_PIPELINE_ADAPTERS['FluxReduxPipeline'],
        'multi_image_reference_edit', main.FLUX_REDUX_REPO, 'hub',
        'c95859fbf7703ca4d6824b4da4407d7cd0434f81')
    bundle._modiff_loaded_stage_artifacts = (
        (main.FLUX_REDUX_REPO, 'c95859fbf7703ca4d6824b4da4407d7cd0434f81'),
        (main.FLUX_DEV_REPO, '3de623fc3c33e44ffbe2bad470d0f45bccf2eb21'))
    bundle._modiff_loaded_stage_offload_mode = 'none'
    bundle._modiff_stage_loading_contract = ('none', 'none', False)
    main._bind_loaded_image_stage(bundle, main.IMAGE_PIPELINE_ADAPTERS['FluxReduxPipeline'], {
        'execution_profile_id': 'flux-redux:direct', 'mode': 'multi_image_reference_edit',
        'pipeline_class': 'FluxReduxPipeline',
        'model_id': {'source': 'hub', 'value': main.FLUX_REDUX_REPO},
        'revision': 'c95859fbf7703ca4d6824b4da4407d7cd0434f81'})
    core.require_owner(bundle, core.PRIOR_IMAGE_STAGE)


def test_joint_serialization_retains_the_requested_basename(actual_tiny_bundle, tmp_path):
    path = serialize_actual_joint_lora(tmp_path, actual_tiny_bundle, 'requested-joint')
    assert path.name == 'requested-joint.safetensors'
