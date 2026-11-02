"""No-weight coverage of the dense Wan 5B image route and retained text identity."""
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.studio_execution_specs import studio_execution_spec_for_pair
from modules.DiffusersVideo import stage_execution as core
from modules.DiffusersVideo.main import LoadPipeline, get_video_pipeline_adapter
from modules.DiffusersVideo.stage_nodes import Denoise


PROFILE = 'wan-22-ti2v-5b:image-direct:v1'
ADAPTER = 'WanTI2VImageToVideoPipeline'
REPO = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'


def test_wan_5b_image_profile_preserves_the_saved_text_route_and_owns_one_task():
    from modiff.auto_resource import AUTO_MODEL_REQUIREMENTS
    text = studio_execution_spec_for_pair('WanTI2VPipeline', 'text_to_video')
    assert text['id'] == 'wan-22-ti2v-5b:text-to-video:v1'
    assert text['contentHash'] == 'studio-spec-v1-a83efd57'
    profile = DIFFUSERS_EXECUTION_PROFILES[PROFILE]
    assert profile.model_type == 'WanTI2VPipeline'
    assert profile.pipeline_class == ADAPTER
    assert profile.modes == ('image_to_video',)
    assert profile.default_repo == REPO
    assert not profile.live_proof
    assert AUTO_MODEL_REQUIREMENTS['WanTI2VPipeline']['supportedTasks'] == ['text_to_video']
    spec = studio_execution_spec_for_pair('WanTI2VPipeline', 'image_to_video')
    assert spec['executionProfileId'] == PROFILE
    assert spec['pipelineClass'] == ADAPTER


def test_wan_5b_image_starter_uses_the_existing_stages_and_exact_single_transformer():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    catalog, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    result = resolve_operation_starter(MODULE_MAP, catalog, {
        'pipelineClass': ADAPTER, 'task': 'image_to_video',
        'executionProfileId': PROFILE, 'authoringRecipeId': 'generic-video-stages:v1',
    })
    assert {node['action'] for node in result['nodes']} == {
        'LoadPipeline', 'EncodeInputs', 'Denoise', 'DecodeLatents',
    }
    loader = next(node for node in result['nodes'] if node['action'] == 'LoadPipeline')
    assert loader['values']['model_id'] == {'source': 'hub', 'value': REPO}
    assert loader['values']['revision'] == require_catalog_revision(REPO)
    assert loader['values']['execution_profile_id'] == PROFILE
    encoder = next(node for node in result['nodes'] if node['action'] == 'EncodeInputs')
    assert encoder['params']['reference_images']['required']
    assert not any('transformer_2' in str(node['values']) for node in result['nodes'])


def test_wan_5b_image_loader_uses_the_official_i2v_class_with_optional_clip_absent():
    adapter = get_video_pipeline_adapter(ADAPTER)
    node = LoadPipeline('wan-5b-image-loader-unit')
    vae, pipeline = object(), SimpleNamespace()
    with (
        patch('diffusers.AutoencoderKLWan.from_pretrained', return_value=vae) as load_vae,
        patch('diffusers.WanImageToVideoPipeline.from_pretrained', return_value=pipeline) as load,
        patch('modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline'),
        patch('modules.DiffusersVideo.main.apply_pipeline_offload'),
        patch.object(node, 'progress'), patch.object(node, 'mm_add'),
    ):
        result = node._load_wan_image_to_video(adapter, {
            'model_id': {'source': 'hub', 'value': REPO},
            'revision': require_catalog_revision(REPO), 'dtype': 'bfloat16',
            'execution_recipe': {'device': 'cpu', 'offload_mode': 'none'},
        })
    assert result is pipeline
    assert load.call_args.args == (REPO,)
    assert load.call_args.kwargs['vae'] is vae
    assert load.call_args.kwargs['revision'] == require_catalog_revision(REPO)
    assert load.call_args.kwargs['image_encoder'] is None
    assert load.call_args.kwargs['image_processor'] is None
    assert load_vae.call_args.kwargs['torch_dtype'] is torch.float32


def image_owner():
    # The official I2V class is shared with A14B, but this loader's selected
    # backend identity and immutable artifact belong only to the dense 5B route.
    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {'input_ids': [1, 2]}

    class Scheduler:
        config = {}

    class WanImageToVideoPipeline:
        _execution_device = 'cpu'
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 16

        def __init__(self):
            self._modiff_video_pipeline_class = ADAPTER
            self._modiff_video_repo = REPO
            self._modiff_video_revision = require_catalog_revision(REPO)
            self.text_encoder = torch.nn.Linear(1, 1)
            self.tokenizer = Tokenizer()
            self.transformer = torch.nn.Linear(1, 1)
            self.transformer.config = SimpleNamespace(patch_size=(1, 2, 2), image_dim=None)
            self.vae = torch.nn.Linear(1, 1)
            self.scheduler = Scheduler()
            self.config = SimpleNamespace(boundary_ratio=None, expand_timesteps=True)
            self.calls = []

        def encode_prompt(self, **kwargs):
            return torch.ones(1, 4, 2), torch.ones(1, 4, 2)

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(frames=torch.ones(1, 1, 3, 2, 2))

    pipeline = WanImageToVideoPipeline()
    core.bind_owner(pipeline, {'execution_profile_id': PROFILE, 'pipeline_class': ADAPTER,
        'model_id': {'source': 'hub', 'value': REPO}, 'revision': require_catalog_revision(REPO)})
    return pipeline


def test_wan_5b_image_denoise_passes_frozen_image_and_embeddings_to_official_latent_call():
    pipeline = image_owner()
    source = Image.new('RGB', (32, 32), 'red')
    encoding = core.encode(pipeline, {'mode': 'image_to_video', 'prompt': 'A boat drifts.',
        'reference_images': [source], 'guidance_scale': 5., 'max_sequence_length': 128})
    node = Denoise('wan-5b-image-stage-unit')
    result = node(pipeline=pipeline, encoded_inputs=encoding, mode='image_to_video',
        width=32, height=32, num_frames=9, guidance_scale=5., num_inference_steps=2,
        scheduler_flow_shift=0., seed=7)
    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert call['image'].tobytes() == source.tobytes()
    assert call['prompt'] is None and call['negative_prompt'] is None
    assert call['prompt_embeds'] is encoding.embeddings['prompt_embeds']
    assert call['output_type'] == 'latent' and call['num_frames'] == 9
    assert 'guidance_scale_2' not in call and 'last_image' not in call
    core.require_latent(pipeline, result['latent_binding'], result['latents_out'])


def test_wan_5b_image_stage_restores_scheduler_and_clears_active_state_after_failure():
    pipeline = image_owner()
    encoding = core.encode(pipeline, {'mode': 'image_to_video', 'prompt': 'A boat drifts.',
        'reference_images': [Image.new('RGB', (32, 32))], 'guidance_scale': 5.})
    node = Denoise('wan-5b-image-failure-unit')
    original, replacement = pipeline.scheduler, type(pipeline.scheduler)()

    def fail(self, **kwargs):
        assert self.scheduler is replacement
        raise RuntimeError('official sampler failure fixture')

    with patch.object(type(pipeline), '__call__', fail), patch(
        'diffusers.UniPCMultistepScheduler.from_config', return_value=replacement,
    ):
        with pytest.raises(RuntimeError, match='official sampler failure fixture'):
            node(pipeline=pipeline, encoded_inputs=encoding, mode='image_to_video',
                width=32, height=32, num_frames=9, guidance_scale=5., scheduler_flow_shift=5., seed=7)
    assert pipeline.scheduler is original
    assert node._active_pipeline is None and node._video_sample is None


def test_reviewed_official_wan_i2v_declares_clip_components_optional():
    import inspect
    from diffusers import WanImageToVideoPipeline
    parameters = inspect.signature(WanImageToVideoPipeline.__init__).parameters
    assert parameters['image_encoder'].default is None
    assert parameters['image_processor'].default is None
    assert parameters['expand_timesteps'].default is False
    assert {'image_encoder', 'image_processor'} <= set(WanImageToVideoPipeline._optional_components)


@pytest.mark.parametrize('references', [None, [], [Image.new('RGB', (32, 32))] * 2])
def test_wan_5b_image_route_rejects_missing_or_multiple_opening_frames(references):
    pipeline = image_owner()
    with pytest.raises(ValueError, match='exactly one opening|source media'):
        core.encode(pipeline, {'mode': 'image_to_video', 'prompt': 'A boat drifts.',
            'reference_images': references, 'guidance_scale': 5.})
    assert pipeline.calls == []
