"""Library opt-outs and source-declared resident audio placement stay authoritative."""
from types import SimpleNamespace

import pytest
import torch

from modiff import diffusers_offload as offload
from modules.ModularDiffusers import modular_utils


class OptOutAudio(torch.nn.Module):
    _supports_group_offloading = False

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(2, 2, 3, padding=1))

    def forward(self, value):
        return self.conv(value)


@pytest.mark.parametrize('pipeline', ['MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline'])
def test_reviewed_audio_placement_preserves_weight_normalized_forward(tmp_path, monkeypatch, pipeline):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(offload, 'supports_accelerator_cpu_offload', lambda device: True)
    config = (modular_utils.MINIMAX_H3_PIPELINE_CONFIG if pipeline == 'MiniMaxH3ModularPipeline'
              else modular_utils.PIKA_H3_FLEX_PIPELINE_CONFIG)
    metadata = modular_utils._pipeline_config_metadata(pipeline, config)
    modes = metadata.get('component_offload_modes', {})
    options = {'resident_components': tuple(name for name, mode in modes.items() if mode == 'none')} if modes else {}
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(43)
        model = OptOutAudio().eval()
        value = torch.randn(1, 2, 8)
        expected = model(value).clone()
        result = offload.apply_component_group_offload(SimpleNamespace(audio_vae=model),
            component_names=['audio_vae'], device='cpu', mode='group_disk', node_id='owned-audio', **options)
        actual = model(value)
        torch.testing.assert_close(actual, expected)
        assert not offload._is_group_offloaded(model)
        assert result.components == [] and result.disk_path is None
        from modules.ModularDiffusers import loaders
        policy = {'component_offload_modes': modes}
        loaders.record_pipeline_component_runtime_policy(SimpleNamespace(components={'audio_vae': model}),
            offload_mode='group_disk', device='cpu', node_id='owned-audio', runtime_policy=policy)
        assert model._modiff_offload_mode == 'none' and model._modiff_offload_node_id is None
        manager = SimpleNamespace(_lookup_ids=lambda **kwargs: ['actual-audio'], get_one=lambda **kwargs: model)
        assert loaders.reusable_component_ids(manager, name='audio_vae', load_id='same-audio', dtype=torch.float32,
            requested_quantization=None, offload_mode='group_disk', device='cpu',
            node_id='different-owner', runtime_policy=policy) == ['actual-audio']
        assert loaders.reusable_component_ids(manager, name='audio_vae', load_id='same-audio', dtype=torch.float32,
            requested_quantization=None, offload_mode='group_disk', device='cpu',
            node_id='different-owner', runtime_policy={}) == []


def test_undeclared_group_offload_never_overrides_the_library_optout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(offload, 'supports_accelerator_cpu_offload', lambda device: True)
    model = OptOutAudio()
    with pytest.raises(RuntimeError, match='does not support group offloading'):
        offload.apply_component_group_offload(SimpleNamespace(audio_vae=model),
            component_names=['audio_vae'], device='cpu', mode='group_disk', node_id='unsupported-audio')
    assert not (tmp_path / 'data/offload').exists()


def test_offload_preflight_uses_the_actual_load_spec_map_without_an_extra_accessor():
    from unittest.mock import Mock
    from modules.ModularDiffusers.loaders import load_components_strict
    load = Mock(side_effect=AssertionError('Weights must not load before the library opt-out check'))
    pipeline = SimpleNamespace(_component_specs={
        'audio_vae': SimpleNamespace(type_hint=OptOutAudio, load=load),
    })
    with pytest.raises(RuntimeError, match='does not support group offloading'):
        load_components_strict(pipeline, names=['audio_vae'], required_names=['audio_vae'],
            model_id='test/unsupported', dtype=torch.float32, offload_mode='group_disk',
            quant_config=None, diagnostics={}, component_load_kwargs={},
            incremental_group_offload={'mode': 'group_disk'})
    load.assert_not_called()


def test_actual_reviewed_audio_vae_preserves_direct_decode_without_group_hooks(tmp_path, monkeypatch):
    from modiff.runtime_overlays import active_diffusers_provider
    provider = active_diffusers_provider()
    if not provider or provider.get('profileId') != 'huggingface-diffusers-video-d0b9b1b':
        pytest.skip('Requires the exact installed reviewed video provider')
    from diffusers import AutoencoderKLMiniMaxH3Audio
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(offload, 'supports_accelerator_cpu_offload', lambda device: True)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(43)
            model = AutoencoderKLMiniMaxH3Audio(encoder_dim=4, encoder_rates=(2,), latent_dim=8,
                latent_channels=2, num_attention_heads=1, decoder_dim=8, decoder_rates=(2,),
                decoder_kernel_sizes=(4,), resblock_kernel_sizes=(3,), resblock_dilation_sizes=((1, 3, 5),)).eval()
            latent = torch.randn(1, 2, 4)
            expected = model.decode(latent, return_dict=False)[0].clone()
            assert model._supports_group_offloading is False
            result = offload.apply_component_group_offload(SimpleNamespace(audio_vae=model),
                component_names=['audio_vae'], device='cpu', mode='group_disk', node_id='actual-audio',
                resident_components=('audio_vae',))
            torch.testing.assert_close(model.decode(latent, return_dict=False)[0], expected)
            assert result.mode == 'none' and not offload._is_group_offloaded(model)
    finally:
        torch.set_num_threads(previous)
