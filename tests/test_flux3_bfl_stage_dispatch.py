"""Joined native stages with real, locally saved tiny BFL components.

The synthetic artifact recipes authorize only these local unit files. Normal
ModelsLoader/ComponentSpec acquisition, the real Qwen3-VL processor and all
native stages execute. No pretrained output, CUDA kernel, fit, Auto, offload
or trained-model LoRA qualification follows from this CPU source audit.
"""
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import save_file

from test_modular_weight_formats import PAIRS, TARGET, artifact, original_state, require_flux3_api


@pytest.fixture
def stage_runtime(tmp_path, monkeypatch):
    require_flux3_api()
    from diffusers import AutoencoderKLFlux2, ComponentSpec, ComponentsManager, Flux3Transformer2DModel
    from diffusers.quantizers.bfl.utils import E2M1_MAX, FP8_MAX, quantize_fp8r, quantize_nvfp4
    from transformers import (Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                              Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor)
    from modiff.config import CONFIG
    from modiff import modular_weight_formats as formats
    import modules.ModularDiffusers as modular
    from modules.ModularDiffusers import denoise, embeddings, latents, loaders

    for field in ('work_dir', 'data', 'images', 'videos', 'audio', 'models', 'upscalers', 'temp'):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    hub = tmp_path / 'hub'
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(hub))
    monkeypatch.setitem(CONFIG.hf, 'online_status', 'Offline')
    manager = ComponentsManager()
    for module in (modular, loaders, denoise, embeddings, latents):
        monkeypatch.setattr(module, 'components', manager)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    nodes = []

    def install(pair, scheme, count):
        model_type, repo, revision = pair
        snapshot = hub / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
        snapshot.mkdir(parents=True)
        fixture = Path(__file__).parent / 'fixtures' / (repo.rsplit('/', 1)[1] + '.modular-index.v1.json')
        (snapshot / 'modular_model_index.json').write_bytes(fixture.read_bytes())
        torch.manual_seed(73)
        transformer = Flux3Transformer2DModel(
            in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
            attention_head_dim=16, num_attention_heads=1, joint_attention_dim=64,
            timestep_channels=8, axes_dims_rope=(4, 4, 4, 4),
            flow_map='Distilled' in model_type,
        ).eval()
        transformer.save_config(snapshot / 'transformer')
        state = original_state(transformer)
        key = 'content_mode_blocks.image.0.q_proj.weight'
        if scheme == 'fp8r':
            state[key], state[key[:-len('.weight')] + '.weight_scale'] = quantize_fp8r(state[key])
        else:
            packed, scale, global_scale = quantize_nvfp4(state[key])
            prefix = key[:-len('.weight')]
            state.update({key: packed, prefix + '.weight_scale': scale,
                          prefix + '.weight_scale_2': global_scale,
                          prefix + '.input_scale': torch.tensor(1.0 / (FP8_MAX * E2M1_MAX))})
        weight = snapshot / 'variants' / scheme / 'model.safetensors'
        weight.parent.mkdir(parents=True)
        save_file(state, weight)
        del transformer, state
        vae = AutoencoderKLFlux2(
            block_out_channels=(4, 4, 4, 4), down_block_types=('DownEncoderBlock2D',) * 4,
            up_block_types=('UpDecoderBlock2D',) * 4, layers_per_block=1,
            latent_channels=2, norm_num_groups=1, mid_block_add_attention=False,
        ).float().eval()
        vae.bn.running_mean.fill_(0.1)
        vae.bn.running_var.fill_(0.8)
        vae.save_pretrained(snapshot / 'vae', safe_serialization=True)
        encoder = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
            text_config={'vocab_size': 16, 'hidden_size': 8, 'intermediate_size': 16,
                         'num_hidden_layers': 32, 'num_attention_heads': 1,
                         'num_key_value_heads': 1, 'head_dim': 8, 'use_cache': False,
                         'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2]}},
            vision_config={'depth': 1, 'hidden_size': 8, 'intermediate_size': 16, 'num_heads': 1,
                           'out_hidden_size': 8, 'patch_size': 2, 'spatial_merge_size': 1,
                           'temporal_patch_size': 1},
            image_token_id=12, video_token_id=13, vision_start_token_id=14, vision_end_token_id=15,
        )).to(dtype=torch.bfloat16).eval()
        encoder.save_pretrained(snapshot / 'text_encoder', safe_serialization=True)
        # Ordinary text tokens cannot collide with the configured vision IDs.
        vocab = {'<|endoftext|>': 0, **{letter: index for index, letter in enumerate('abcdefghijk', 1)},
                 '<|image_pad|>': 12, '<|video_pad|>': 13,
                 '<|vision_start|>': 14, '<|vision_end|>': 15}
        tokenizer = Qwen2TokenizerFast(vocab=vocab, merges=[], pad_token='<|endoftext|>',
                                      additional_special_tokens=list(vocab)[12:])
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(), chat_template="{{ messages[0]['content'] }}",
        )
        processor.save_pretrained(snapshot / 'text_encoder')
        assert processor.tokenizer('abc')['input_ids'] == [1, 2, 3]
        assert processor.image_token_id == 12 and processor.video_token_id == 13
        row = {'id': model_type + ':' + scheme + ':joined-synthetic-unit-v1', 'status': 'contract_only',
               'modelType': model_type, 'repository': repo, 'revision': revision,
               'weightFormat': scheme, 'scheme': scheme, 'loaderLibrary': 'diffusers',
               'loaderClass': 'Flux3Transformer2DModel',
               'weight': artifact(weight, weight.relative_to(snapshot).as_posix()),
               'config': artifact(snapshot / 'transformer/config.json', 'transformer/config.json'),
               'downloadFiles': sorted(path.relative_to(snapshot).as_posix()
                                       for path in snapshot.rglob('*') if path.is_file())}
        recipes = tmp_path / 'joined-synthetic-unit-recipes.json'
        recipes.write_text(json.dumps({'schemaVersion': 1, 'recipes': [row]}), encoding='utf-8')
        monkeypatch.setattr(formats, 'RECIPES_PATH', recipes)
        del vae, encoder, processor
        acquired = []
        actual_load = ComponentSpec.load

        def capture_load(spec, **kwargs):
            assert spec.name != 'scheduler', 'A default-created scheduler must not enter pretrained acquisition'
            acquired.append(spec.name)
            return actual_load(spec, **kwargs)

        monkeypatch.setattr(ComponentSpec, 'load', capture_load)
        node = loaders.ModelsLoader('joined-models')
        nodes.append(node)
        values = dict(model_type=model_type, repo_id={'source': 'hub', 'value': repo}, revision=revision,
                      workflow_id='image_conditioned' if count else 'text2image', dtype='bfloat16', device='cpu',
                      weight_format=scheme, offload_mode='none', auto_offload=False, weight_loading_mode='cpu',
                      trust_remote_code=False, quant_config=None, lora_list=None, unet=None, vae=None, controlnet=None)
        node(**values)
        assert isinstance(node.loader.tokenizer, Qwen3VLProcessor)
        assert node.loader.tokenizer.tokenizer('abc')['input_ids'] == [1, 2, 3]
        assert node.loader.tokenizer.apply_chat_template([{'role': 'user', 'content': 'abc'}],
                                                        tokenize=False, add_generation_prompt=True) == 'abc'
        return SimpleNamespace(node=node, owner=node.loader, payload=node.output, manager=manager,
                               snapshot=snapshot, values=values, weight=weight, acquired=acquired)

    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            yield install
    finally:
        for node in nodes:
            node.loader = None
            node.output.clear()
        manager.components.clear()
        manager.collections.clear()
        gc.collect()
        torch.set_num_threads(threads)


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
def test_actual_loader_publishes_owned_default_created_scheduler(stage_runtime, pair):
    from modules.ModularDiffusers.route_state import require_component_binding
    loaded = stage_runtime(pair, 'fp8r', 0)
    spec = loaded.owner.get_component_spec('scheduler')
    assert spec.default_creation_method == 'from_config'
    assert type(loaded.owner.scheduler) is spec.type_hint
    assert sorted(loaded.acquired) == ['text_encoder', 'tokenizer', 'transformer', 'vae']
    output = loaded.payload['scheduler']
    assert loaded.manager.get_one(output['model_id']) is loaded.owner.scheduler
    assert loaded.manager._lookup_ids(name='scheduler', collection='joined-models') == {output['model_id']}
    token = require_component_binding(loaded.payload['unet_out'], label='denoiser',
                                      expected_model_type=pair[0], expected_role='denoiser')
    assert require_component_binding(output, label='scheduler', expected_model_type=pair[0],
                                     expected_token=token, expected_role='scheduler') is token
    assert loaded.owner.scheduler.config.shift == 6.93


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
@pytest.mark.parametrize('reference_count', [0, 2])
def test_real_single_file_owner_runs_joined_stages(stage_runtime, pair, scheme, reference_count):
    import diffusers
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.quantizers.bfl.utils import BFLFP8RLinear, BFLNVFP4Linear
    from modiff.NodeBase import NodeBase
    from modules.ModularDiffusers.embeddings import EncodePrompt
    from modules.ModularDiffusers.denoise import Denoise
    from modules.ModularDiffusers.latents import DecodeLatents, ImageEncode

    loaded = stage_runtime(pair, scheme, reference_count)
    owner, payload = loaded.owner, loaded.payload
    model = owner.transformer
    layer = model.get_submodule(TARGET)
    assert isinstance(layer, BFLFP8RLinear if scheme == 'fp8r' else BFLNVFP4Linear)
    assert layer.compute_dtype == torch.bfloat16 and owner.text_encoder.dtype == torch.bfloat16
    assert owner.vae.dtype == torch.float32
    assert not (loaded.snapshot / 'transformer/diffusion_pytorch_model.safetensors').exists()
    selected = loaded.node._weight_format_binding
    assert selected is owner._modiff_weight_format_binding
    before_state = {name: tensor.detach().clone() for name, tensor in layer.state_dict().items()}
    before_bn = (owner.vae.bn.running_mean.clone(), owner.vae.bn.running_var.clone())
    encode, image_encode = EncodePrompt('joined-prompt'), ImageEncode('joined-reference')
    sampler, decode = Denoise('joined-denoise'), DecodeLatents('joined-decode')
    progress = []
    sampler.progress = lambda percentage, **fields: progress.append((percentage, fields))
    assert isinstance(loaded.node, NodeBase) and loaded.node._cache_valid
    assert all(type(node).__call__ is NodeBase.__call__ for node in (encode, image_encode, sampler, decode))
    refs = [Image.new('RGB', (64, 48), (220, 40, 20)), Image.new('RGB', (32, 32), (10, 80, 210))][:reference_count]
    image_values = image_encode(vae=payload['vae_out'], image=refs) if refs else None
    encode_values = dict(text_encoders=payload['text_encoders'], prompt='abc', max_sequence_length=8,
                         text_encoder_out_layers=(4, 8, 12, 16, 20, 24, 28, 32))
    if pair[0] == 'Flux3ModularPipeline':
        encode_values['negative_prompt'] = 'def'
    if image_values:
        assert len(image_values['image_cond_latents']) == 2
        assert [tuple(tensor.shape[-2:]) for tensor in image_values['image_cond_latents']] == [(3, 4), (2, 2)]
        assert all(tensor.dtype == torch.float32 for tensor in image_values['image_cond_latents'])
        encode_values['reference_binding'] = image_values['reference_binding']
    encoded = encode(**encode_values)
    assert encoded['embeddings']['prompt_embeds'].shape == (1, 8, 64)
    assert encoded['embeddings']['prompt_embeds'].dtype == torch.bfloat16
    if pair[0] == 'Flux3ModularPipeline':
        assert encoded['embeddings']['negative_prompt_embeds'] is not None
    else:
        assert 'negative_prompt_embeds' not in encoded['embeddings']
    sampler_values = dict(unet=payload['unet_out'], scheduler=payload['scheduler'],
                          embeddings=encoded['embeddings'], seed=91, width=64, height=64,
                          num_inference_steps=2, num_images_per_prompt=1)
    if image_values:
        sampler_values.update(image_cond_latents=image_values['image_cond_latents'],
                              reference_binding=image_values['reference_binding'])
    denoised = sampler(**sampler_values)
    expected_latents = denoised['latents'].clone()
    assert expected_latents.shape == (1, 8, 4, 4)
    assert expected_latents.dtype == torch.bfloat16 and torch.isfinite(expected_latents).all()
    assert [fields['current_step'] for _percentage, fields in progress
            if fields.get('phase') == 'denoising'] == [0, 1, 2]
    decode_dtypes = []
    handle = owner.vae.decoder.register_forward_pre_hook(lambda _module, args: decode_dtypes.append(args[0].dtype))
    try:
        decoded = decode(vae=payload['vae_out'], latents=denoised['latents'])
    finally:
        handle.remove()
    assert decode_dtypes == [torch.float32]
    image = decoded['images']
    if isinstance(image, list):
        assert len(image) == 1
        image = image[0]
    assert isinstance(image, Image.Image) and image.size == (64, 64)
    assert np.isfinite(np.asarray(image)).all()
    assert encode._pipeline.text_encoder is owner.text_encoder
    assert encode._pipeline.tokenizer is owner.tokenizer
    assert sampler._pipeline.transformer is model and decode._pipeline.vae is owner.vae

    # Compare identical local components to the official selected whole tree;
    # this is API/stage parity, not trained-media quality or dense-model parity.
    workflow = 'image_conditioned' if refs else 'text2image'
    whole = getattr(diffusers, pair[0])(workflow=workflow)
    whole.update_components(text_encoder=owner.text_encoder, tokenizer=owner.tokenizer,
                            transformer=model, vae=owner.vae,
                            scheduler=FlowMatchEulerDiscreteScheduler(shift=6.93))
    values = dict(prompt='abc', max_sequence_length=8, text_encoder_out_layers=(4, 8, 12, 16, 20, 24, 28, 32),
                  width=64, height=64, num_inference_steps=2, num_images_per_prompt=1,
                  generator=torch.Generator(device='cpu').manual_seed(91))
    if pair[0] == 'Flux3ModularPipeline':
        values['negative_prompt'] = 'def'
    if refs:
        values['image'] = refs
    direct = whole(**values, output=['latents', 'images'])
    torch.testing.assert_close(direct['latents'], expected_latents, rtol=0, atol=0)
    direct_image = direct['images'][0] if isinstance(direct['images'], list) else direct['images']
    np.testing.assert_array_equal(np.asarray(image), np.asarray(direct_image))
    encode(**encode_values)
    sampler(**sampler_values)
    assert not encode._has_changed and not sampler._has_changed
    sampler.invalidate_cache()
    torch.testing.assert_close(sampler(**sampler_values)['latents'], expected_latents, rtol=0, atol=0)
    assert sampler._has_changed
    changed_seed = sampler(**{**sampler_values, 'seed': 92})
    assert sampler._has_changed and not torch.equal(changed_seed['latents'], expected_latents)
    selected.assert_current()
    assert torch.equal(owner.vae.bn.running_mean, before_bn[0])
    assert torch.equal(owner.vae.bn.running_var, before_bn[1])
    for name, tensor in layer.state_dict().items():
        assert torch.equal(tensor.contiguous().reshape(-1).view(torch.uint8),
                           before_state[name].contiguous().reshape(-1).view(torch.uint8))


@pytest.mark.parametrize('fault', ['missing', 'wrong_class', 'unbound', 'pretrained'])
def test_default_scheduler_publication_rejects_unbound_or_wrong_objects(fault):
    from diffusers import ComponentSpec, ComponentsManager, DDPMScheduler, FlowMatchEulerDiscreteScheduler
    from modules.ModularDiffusers.loaders import _register_default_created_scheduler
    spec = ComponentSpec(name='scheduler', type_hint=FlowMatchEulerDiscreteScheduler,
                         default_creation_method='from_pretrained' if fault == 'pretrained' else 'from_config')
    scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93)
    if fault == 'missing':
        scheduler = None
    elif fault == 'wrong_class':
        scheduler = DDPMScheduler()
    pipeline = SimpleNamespace(_component_specs={} if fault == 'unbound' else {'scheduler': spec},
                               scheduler=scheduler)
    manager = ComponentsManager()
    if fault in ('missing', 'wrong_class'):
        with pytest.raises(ValueError, match='exact reviewed class'):
            _register_default_created_scheduler(pipeline, manager, 'owner')
    else:
        assert _register_default_created_scheduler(pipeline, manager, 'owner') is False
    assert not manager.components and not manager.collections


def test_default_scheduler_publication_preserves_other_owners_during_replacement():
    from diffusers import ComponentSpec, ComponentsManager, FlowMatchEulerDiscreteScheduler
    from modules.ModularDiffusers.loaders import _register_default_created_scheduler
    spec = ComponentSpec(name='scheduler', type_hint=FlowMatchEulerDiscreteScheduler,
                         default_creation_method='from_config')
    old = FlowMatchEulerDiscreteScheduler(shift=6.93)
    manager = ComponentsManager()
    old_id = manager.add('scheduler', old, collection='owner')
    manager.add('scheduler', old, collection='surviving-owner')
    current = FlowMatchEulerDiscreteScheduler(shift=6.93)
    pipeline = SimpleNamespace(_component_specs={'scheduler': spec}, scheduler=current)
    assert _register_default_created_scheduler(pipeline, manager, 'owner') is True
    current_ids = manager._lookup_ids(name='scheduler', collection='owner')
    assert len(current_ids) == 1 and manager.get_one(next(iter(current_ids))) is current
    assert manager._lookup_ids(name='scheduler', collection='surviving-owner') == {old_id}
    assert manager.get_one(old_id) is old
    assert _register_default_created_scheduler(pipeline, manager, 'owner') is True
    assert manager._lookup_ids(name='scheduler', collection='owner') == current_ids
