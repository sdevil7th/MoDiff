from unittest.mock import Mock

import pytest
import torch
from diffusers import AutoencoderTiny, DreamLiteMobilePipeline, DreamLiteUNetModel, FlowMatchEulerDiscreteScheduler
from PIL import Image

from modiff.model_artifact_catalog import require_catalog_revision
from modules.DiffusersImage import main as image


@pytest.mark.parametrize(('mode', 'tiling'), [('edit_image', True), ('edit_image', False), ('text_to_image', True)])
def test_dreamlite_mobile_tiled_edit_matches_bfloat16_unet(monkeypatch, mode, tiling):
    class Encoder(torch.nn.Linear):
        @property
        def dtype(self):
            return self.weight.dtype

        @property
        def device(self):
            return self.weight.device

    dtype = torch.bfloat16
    vae = AutoencoderTiny(
        encoder_block_out_channels=(4, 4, 4, 4), decoder_block_out_channels=(4, 4, 4, 4),
        num_encoder_blocks=(1, 1, 1, 1), num_decoder_blocks=(1, 1, 1, 1),
    ).to(dtype)
    vae.tile_sample_min_size = 32
    vae.tile_latent_min_size = 4
    vae.tile_overlap_factor = 0.5
    unet = DreamLiteUNetModel(
        down_block_types=('DreamLiteDownBlock2D',), up_block_types=('DreamLiteUpBlock2D',),
        block_out_channels=(8,), layers_per_block=1, norm_num_groups=4,
        attention_head_dim=2, cross_attention_dim=8, mid_block_type=None,
    ).to(dtype)
    pipeline = DreamLiteMobilePipeline(Encoder(2, 2), None, None, vae, unet, FlowMatchEulerDiscreteScheduler())
    upstream_prepare = pipeline.prepare_image_latents
    load = Mock(return_value=pipeline)
    offload = Mock()
    monkeypatch.setattr(image, 'load_cached_image_component', load)
    monkeypatch.setattr(image, 'apply_pipeline_offload', offload)
    node = image.LoadPipeline('dreamlite-mobile-tiled-edit')
    node.progress = Mock()
    node.mm_add = Mock()
    execution_recipe = dict(device='cuda:0', offload_mode='model_cpu', vae_slicing=True, vae_tiling=tiling)
    loaded = node.execute(
        model_id=image.DREAMLITE_MOBILE_REPO, pipeline_class='DreamLiteMobilePipeline', mode=mode,
        revision=require_catalog_revision(image.DREAMLITE_MOBILE_REPO), dtype='bfloat16',
        execution_recipe=execution_recipe,
    )['pipeline']

    assert loaded is pipeline
    assert load.call_args.kwargs['torch_dtype'] == dtype
    assert pipeline.vae.use_tiling is tiling
    assert pipeline.vae.use_slicing
    assert offload.call_args.args == (pipeline,)
    assert offload.call_args.kwargs['mode'] == 'model_cpu'
    assert offload.call_args.kwargs['device'] == 'cuda:0'
    assert pipeline.text_encoder.dtype == unet.time_embedding.linear_1.weight.dtype == dtype

    device = torch.device('cpu')
    source = Image.new('RGB', (32, 32))
    with torch.no_grad():
        raw_image_latents = upstream_prepare(pipeline.image_processor.preprocess(source), dtype=dtype, device=device)
        assert raw_image_latents.dtype == (torch.float32 if tiling else dtype)

    def encode_prompt(*args, dtype, device, **kwargs):
        # Bypass tokenization and text weights. Real Diffusers runs VAE and UNet below.
        return torch.zeros((1, 1, 8), dtype=dtype, device=device), torch.ones((1, 1), device=device)

    monkeypatch.setattr(pipeline, 'encode_prompt', encode_prompt)
    samples = []
    with unet.register_forward_pre_hook(lambda _, args: samples.append(args[0])):
        action = image.Edit('dreamlite-mobile-edit') if mode == 'edit_image' else image.Generate('dreamlite-mobile-generate')
        action.progress = Mock()
        result = action.execute(
            pipeline=loaded, prompt='Offline fixture', seed=42, width=32, height=32,
            num_inference_steps=1, guidance_scale=0, output_type='pt',
            **({'image': source} if mode == 'edit_image' else {}),
        )
    assert len(samples) == 1
    sample = samples[0]
    assert sample.dtype == dtype
    conditioning = sample[..., sample.shape[-1] // 2:]
    torch.testing.assert_close(conditioning, raw_image_latents.to(dtype=dtype) if mode == 'edit_image'
                               else torch.zeros_like(conditioning))
    assert result['images'].shape == (1, 3, 32, 32)
    assert torch.isfinite(result['images']).all()
    assert pipeline.prepare_image_latents == upstream_prepare
    assert DreamLiteMobilePipeline.prepare_image_latents is upstream_prepare.__func__
