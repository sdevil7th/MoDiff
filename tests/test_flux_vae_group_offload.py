"""Real direct VAE entry points must work through pipeline-level group offload."""
import os
from unittest.mock import patch

import pytest
import torch


@pytest.mark.parametrize('mode', ['group_cpu', 'group_disk'])
def test_pipeline_group_offload_keeps_vae_on_its_leaf_hook_path(mode, tmp_path):
    from modiff.diffusers_offload import apply_pipeline_offload
    class Pipeline:
        vae = torch.nn.Conv2d(2, 2, 1)
        transformer = torch.nn.Linear(2, 2)
        def enable_group_offload(self, **kwargs):
            assert kwargs['exclude_modules'] == ['vae']
    calls = []
    def record_module(module, **kwargs):
        if isinstance(module, torch.nn.Module):
            calls.append(kwargs)
            return kwargs['component_name']
    with patch('modiff.diffusers_offload._apply_group_to_module',
               side_effect=record_module), \
         patch('modiff.diffusers_offload._disk_path', return_value=tmp_path), \
         patch('modiff.diffusers_offload._streaming_offload_allowed', return_value=False):
        result = apply_pipeline_offload(Pipeline(), mode=mode, device='cuda:0', node_id='test')
    assert result.applied
    assert len(calls) == (1 if mode == 'group_cpu' else 2)
    vae_calls = [call for call in calls if call['component_name'] == 'vae']
    assert len(vae_calls) == 1
    assert vae_calls[0]['offload_type'] == 'leaf_level'


@pytest.mark.skipif(os.environ.get('MODIFF_TEST_FLUX_VAE_ACCELERATOR') != '1',
                    reason='Explicit accelerator regression; no downloads')
@pytest.mark.parametrize('mode', ['group_cpu', 'group_disk'])
@pytest.mark.parametrize('vae_class_name', ['AutoencoderKL', 'AutoencoderKLFlux2'])
def test_actual_flux_vae_encode_decode_under_pipeline_group_offload(mode, vae_class_name, tmp_path):
    import diffusers
    from diffusers import DiffusionPipeline
    from modiff.diffusers_offload import apply_pipeline_offload
    class Pipeline(DiffusionPipeline):
        def __init__(self, vae):
            super().__init__()
            self.register_modules(vae=vae)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        vae = getattr(diffusers, vae_class_name)(block_out_channels=(32,)*4, layers_per_block=1,
            latent_channels=2, down_block_types=('DownEncoderBlock2D',)*4,
            up_block_types=('UpDecoderBlock2D',)*4).to(dtype=torch.bfloat16).eval()
        pixels = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16, device='cuda:0')
        vae.to('cuda:0')
        with torch.no_grad():
            reference_latents = vae.encode(pixels).latent_dist.mode()
            reference_pixels = vae.decode(reference_latents).sample
        vae.to('cpu')
        buffers_before = {name: value.clone() for name, value in vae.named_buffers()}
        pipeline = Pipeline(vae)
        with patch('modiff.diffusers_offload._disk_path', return_value=tmp_path), \
             patch('modiff.diffusers_offload._streaming_offload_allowed', return_value=False):
            apply_pipeline_offload(pipeline, mode=mode, device='cuda:0', node_id='test')
        # FLUX.2 reads BatchNorm statistics directly outside VAE.forward before
        # both encode/decode. Disk hooks must not replace them with empty data.
        for name, value in vae.named_buffers():
            torch.testing.assert_close(value.cpu(), buffers_before[name], rtol=0, atol=0)
        with torch.no_grad():
            for _ in range(2):
                latent = vae.encode(pixels).latent_dist.mode()
                decoded = vae.decode(latent).sample
                torch.testing.assert_close(latent, reference_latents, rtol=0, atol=0)
                torch.testing.assert_close(decoded, reference_pixels, rtol=0, atol=0)
    finally:
        torch.set_num_threads(old_threads)
