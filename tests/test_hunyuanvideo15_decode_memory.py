"""Decode working memory is explicit and never rewrites a shared VAE owner."""
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest

from modules.ModularDiffusers import workflow_blocks as blocks


@pytest.mark.parametrize('initial', [False, True])
@pytest.mark.parametrize('selected', [None, False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_decoder_applies_and_restores_official_tiling_hooks(initial, selected, fails):
    from diffusers import AutoencoderKLHunyuanVideo15
    vae = AutoencoderKLHunyuanVideo15(block_out_channels=(32, 32, 32, 32, 32), layers_per_block=1)
    if initial:
        vae.enable_tiling()
    observed = []

    class Pipeline:
        def __init__(self):
            self.vae = vae

        def __call__(self, **kwargs):
            observed.append(vae.use_tiling)
            if fails:
                raise RuntimeError('decode allocation fault')
            return {'videos': [[Image.new('RGB', (16, 16))]]}

    node = blocks.WorkflowHunyuanVideo15Decode()
    pipeline = Pipeline()
    with patch.object(blocks, '_hunyuan_video_15_state',
                      return_value=('HunyuanVideo15ModularPipeline', 'text2video', object(), pipeline, {})):
        values = {} if selected is None else {'vae_tiling': selected}
        if fails:
            with pytest.raises(RuntimeError, match='decode allocation fault'):
                node.execute(**values)
        else:
            result = node.execute(**values)
            assert len(result['video']) == 1
            assert node._execution_input_record['fields']['vae_tiling']['value'] is (
                initial if selected is None else selected)
    assert observed == [initial if selected is None else selected]
    assert vae.use_tiling is initial


@pytest.mark.parametrize('value', ['true', 1, {}, []])
def test_decoder_rejects_invalid_memory_control_before_call(value):
    node = blocks.WorkflowHunyuanVideo15Decode()
    with patch.object(blocks, '_hunyuan_video_15_state') as prepare:
        with pytest.raises(ValueError, match='boolean'):
            node.execute(vae_tiling=value)
    prepare.assert_not_called()


def test_saved_legacy_decoder_keeps_inherit_while_fresh_720p_defaults_are_tiled():
    from modiff.hunyuan_video15_artifacts import HUNYUAN_720P_T2V_REPOSITORY
    from modiff.studio_execution_specs import studio_capability_definition
    from modiff.hunyuan_video15_artifacts import stage_authoring_defaults
    assert blocks.WorkflowHunyuanVideo15Decode.params['vae_tiling']['default'] is None
    capability = studio_capability_definition('HunyuanVideo15ModularPipeline', repository=HUNYUAN_720P_T2V_REPOSITORY)
    values = stage_authoring_defaults('WorkflowHunyuanVideo15Decode', 'text2video', {}, capability)
    assert values['vae_tiling'] is True


def test_missing_vae_memory_contract_is_not_silently_accepted():
    node = blocks.WorkflowHunyuanVideo15Decode()
    pipeline = SimpleNamespace(vae=object())
    with patch.object(blocks, '_hunyuan_video_15_state', return_value=('', '', None, pipeline, {})):
        with pytest.raises(ValueError, match='tiling'):
            node.execute(vae_tiling=True)


def test_official_decode_uses_real_tiles_and_restores_policy_after_native_tile_fault(monkeypatch):
    import torch
    from diffusers import AutoencoderKLHunyuanVideo15, HunyuanVideo15ModularPipeline
    from diffusers.modular_pipelines import PipelineState

    definition = HunyuanVideo15ModularPipeline()
    block = blocks._official_stage_block(definition.blocks.get_workflow('text2video'), 'decode')
    pipeline = block.init_pipeline()
    vae = AutoencoderKLHunyuanVideo15(block_out_channels=(32,) * 5, layers_per_block=1).eval()
    vae.enable_tiling(tile_sample_min_height=32, tile_sample_min_width=32,
                      tile_latent_min_height=2, tile_latent_min_width=2, tile_overlap_factor=.5)
    vae.disable_tiling()
    pipeline.update_components(vae=vae)
    node = blocks.WorkflowHunyuanVideo15Decode()
    latents = torch.zeros((1, 32, 2, 2, 3))
    tile_calls = []
    native_decode = vae.decoder.forward

    def run():
        state = PipelineState()
        state.set('latents', latents)
        return ('HunyuanVideo15ModularPipeline', 'text2video', object(), pipeline, state)

    def observe_tile(tile):
        tile_calls.append(tuple(tile.shape))
        assert vae.use_tiling
        return native_decode(tile)

    # Only the already separately tested loader/state envelope is supplied
    # here. Both the selected Diffusers decoder stage and its VAE are real.
    monkeypatch.setattr(blocks, '_hunyuan_video_15_state', lambda *args: run())
    monkeypatch.setattr(vae.decoder, 'forward', observe_tile)
    frames = node.execute(vae_tiling=True)['video']
    assert len(frames) == 5 and all(frame.size == (48, 32) for frame in frames)
    assert len(tile_calls) > 1 and all(shape[-1] <= 2 for shape in tile_calls)
    assert vae.use_tiling is False

    def fail_in_native_tile(tile):
        assert vae.use_tiling and tile.shape[-1] <= 2
        raise torch.OutOfMemoryError('controlled failure during a native decode tile')

    monkeypatch.setattr(vae.decoder, 'forward', fail_in_native_tile)
    with pytest.raises(torch.OutOfMemoryError, match='native decode tile'):
        node.execute(vae_tiling=True)
    assert vae.use_tiling is False


@pytest.mark.parametrize('profile,expected', [
    ('hunyuan-video15-720p-t2v:official-modular-workflow', True),
    ('hunyuan-video15-720p-i2v:official-modular-workflow', True),
    ('hunyuan-video15-t2v:official-modular-workflow', None),
])
def test_fresh_shared_starter_exposes_the_exact_decoder_memory_choice(profile, expected):
    from modules import MODULE_MAP
    from modiff.diffusers_profiles import public_execution_profiles
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    task = 'image_to_video' if 'i2v' in profile else 'text_to_video'
    contracts, _ = build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})
    result = resolve_operation_starter(MODULE_MAP, contracts, {
        'pipelineClass': 'HunyuanVideo15ModularPipeline', 'task': task, 'executionProfileId': profile,
    })
    decoder = next(node for node in result['nodes'] if node['action'] == 'WorkflowHunyuanVideo15Decode')
    assert decoder['values'].get('vae_tiling') is expected
    assert decoder['params']['vae_tiling'].get('value', decoder['params']['vae_tiling']['default']) is expected
    assert not decoder['params']['vae_tiling'].get('hidden')
