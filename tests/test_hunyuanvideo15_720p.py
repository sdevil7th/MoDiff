"""720p selects real checkpoint contracts without changing the 480p recipes."""
from unittest.mock import patch
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.modular_workflow_contracts import (
    effective_reviewed_modular_selector, reviewed_modular_workflow_variants,
    require_reviewed_modular_repository_workflow,
)
from modiff.studio_execution_specs import studio_capability_definition, studio_execution_profile_definitions
from modules.ModularDiffusers.route_state import issue_pipeline_instance_token
from modules.ModularDiffusers.workflow_blocks import WorkflowHunyuanVideo15Denoise, _issue_workflow_state

MODEL = 'HunyuanVideo15ModularPipeline'
BASE_T2V = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v'
BASE_I2V = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled'
T2V = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v'
I2V = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v'
T2V_PIN = 'f4dbc4a1efa4ac8ea56680cdf79d9f455105e814'
I2V_PIN = 'dcf088ef4e420dd54bc25930ee810bb6613dfcb5'


@pytest.mark.parametrize('workflow,stage,dependency', [
    ('text2video', 'text_encoder', 'transformer'),
    ('image2video', 'text_encoder', 'transformer'),
    ('image2video', 'vae_encoder', 'transformer'),
    ('text2video', 'denoise', 'vae'),
    ('image2video', 'denoise', 'vae'),
])
def test_isolated_native_hunyuan_stages_retain_cross_stage_component_dependencies(workflow, stage, dependency):
    from diffusers import HunyuanVideo15ModularPipeline
    from modules.ModularDiffusers.workflow_blocks import (
        _official_stage_block, _official_stage_component_dependencies,
    )

    definition = HunyuanVideo15ModularPipeline()
    block = _official_stage_block(definition.blocks.get_workflow(workflow), stage)
    selected = _official_stage_component_dependencies(
        block, pipeline_class=MODEL, block_path=stage, definition=definition,
    )
    # Use the installed official block and pipeline constructor, with no weights.
    # The live encoder reads transformer.dtype; VAE encoding uses target_size;
    # latent preparation reads VAE geometry through pipeline properties.
    pipeline = selected.init_pipeline()
    assert dependency in pipeline.pretrained_component_names
    expected = {spec.name: spec for spec in definition.blocks.expected_components}
    selected_specs = {spec.name: spec for spec in selected.expected_components}
    assert selected_specs[dependency] == expected[dependency]


@pytest.mark.parametrize('kind,repository,revision,blob', [
    ('t2v', T2V, T2V_PIN, 'ff0a2bcb99c7b1fe1f7eb534ad98621c731003bb'),
    ('i2v', I2V, I2V_PIN, 'e910e64667e63d20665fc7b6d2df472f112505d1'),
])
@pytest.mark.parametrize('fault', [None, 'pipeline', 'component', 'null_required', 'extra'])
def test_exact_standard_indexes_match_the_official_installed_component_contract(kind, repository, revision, blob, fault):
    from modules.ModularDiffusers.loaders import _validate_reviewed_pipeline_index

    raw = (Path(__file__).parent / 'fixtures' / f'hunyuanvideo15_720p_{kind}_index.v1.json').read_bytes()
    assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob
    document = deepcopy(json.loads(raw))
    if fault == 'pipeline':
        document['_class_name'] = 'MiniMaxH3ModularPipeline'
    elif fault == 'component':
        document['transformer'] = ['diffusers', 'WanTransformer3DModel']
    elif fault == 'null_required':
        document['text_encoder_2'] = [None, None]
    elif fault == 'extra':
        document['foreign_encoder'] = ['transformers', 'T5EncoderModel']
    with patch('modules.ModularDiffusers.loaders._load_reviewed_pipeline_index',
               return_value=('model_index.json', document)):
        if fault:
            with pytest.raises(ValueError):
                _validate_reviewed_pipeline_index(MODEL, repository, revision)
        else:
            name, result = _validate_reviewed_pipeline_index(MODEL, repository, revision)
            assert name == 'model_index.json' and result['transformer'][1] == 'HunyuanVideo15Transformer3DModel'


@pytest.mark.parametrize('repo,revision,workflow,baseline', [
    (T2V, T2V_PIN, 'text2video', BASE_T2V), (I2V, I2V_PIN, 'image2video', BASE_I2V),
])
def test_720p_exact_variant_selection_retains_same_generic_workflow(repo, revision, workflow, baseline):
    pin = catalog_repository_pin(repo, model_type=MODEL)
    assert pin is not None and pin['revision'] == revision
    assert reviewed_modular_workflow_variants(model_type=MODEL, workflow_id=workflow,
        default_repository=baseline) == (baseline, repo)
    selected, selected_pin = effective_reviewed_modular_selector(model_type=MODEL,
        repo_id={'source': 'hub', 'value': baseline}, revision=None,
        workflow_id=workflow, reviewed_variant=repo, default_repository=baseline)
    assert selected == {'source': 'hub', 'value': repo} and selected_pin == revision
    require_reviewed_modular_repository_workflow(MODEL, repo, revision, workflow)
    for bad in (None, 'main', 'a' * 40):
        with pytest.raises(ValueError, match='workflow/revision'):
            require_reviewed_modular_repository_workflow(MODEL, repo, bad, workflow)
    with pytest.raises(ValueError, match='workflow/revision'):
        require_reviewed_modular_repository_workflow(MODEL, repo, revision,
            'image2video' if workflow == 'text2video' else 'text2video')


@pytest.mark.parametrize('repo,mode,profile_id', [
    (T2V, 'text_to_video', 'hunyuan-video15-720p-t2v:official-modular-workflow'),
    (I2V, 'image_to_video', 'hunyuan-video15-720p-i2v:official-modular-workflow'),
])
def test_720p_profiles_expose_full_sampling_without_promoting_qualification(repo, mode, profile_id):
    profile = studio_execution_profile_definitions()[profile_id]
    assert profile['model_type'] == profile['pipeline_class'] == MODEL
    assert profile['default_repo'] == repo and profile['modes'] == (mode,)
    assert profile['supported_offload_modes'] == ('none', 'model_cpu')
    capability = studio_capability_definition(MODEL, repository=repo)
    assert capability['defaultRepo'] == repo
    assert capability['recommendedSteps'] == 50 and capability['recommendedGuidance'] == 6.0
    assert capability['defaultSize']['width'] == 1280 and capability['defaultSize']['height'] == 720
    assert capability['modeDefaults'][mode] == {'steps': 50, 'guidanceScale': 6.0}
    assert not any(capability[name] for name in ('autoEligible', 'templateEligible', 'galleryEligible', 'liveProof'))
    assert studio_capability_definition(MODEL)['defaultRepo'] == BASE_T2V
    assert studio_capability_definition(MODEL)['modeDefaults']['image_to_video'] == {'steps': 12, 'guidanceScale': 1.0}


class Guider:
    def __init__(self, guidance_scale=6.0):
        self.guidance_scale = guidance_scale

    def new(self, *, guidance_scale):
        return Guider(guidance_scale)


@pytest.mark.parametrize('profile_id,repo,steps,guidance', [
    ('hunyuan-video15-i2v-step-distilled:official-modular-workflow', BASE_I2V, 12, 1.0),
    ('hunyuan-video15-720p-i2v:official-modular-workflow', I2V, 50, 6.0),
])
def test_fresh_i2v_starter_does_not_inject_duplicate_geometry_into_denoise(profile_id, repo, steps, guidance):
    from modules import MODULE_MAP
    from modiff.diffusers_profiles import public_execution_profiles
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter

    contracts, _ = build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})
    result = resolve_operation_starter(MODULE_MAP, contracts, {'pipelineClass': MODEL,
        'task': 'image_to_video', 'executionProfileId': profile_id})
    nodes = {node['action']: node for node in result['nodes']}
    assert nodes['ModelsLoader']['values']['repo_id'] == {'source': 'hub', 'value': repo}
    denoise = nodes['WorkflowHunyuanVideo15Denoise']
    assert denoise['values']['num_inference_steps'] == steps
    assert denoise['values']['guidance_scale'] == guidance
    for field in ('width', 'height'):
        assert denoise['values'].get(field) is None
        assert denoise['params'][field].get('value') is None


class Pipeline:
    def __init__(self):
        self.guider = Guider()
        self.calls = []

    def update_components(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {'latents': object()}


@pytest.mark.parametrize('repo,revision,steps,guidance,allowed', [
    (I2V, I2V_PIN, 50, 6.0, True),
    (I2V, I2V_PIN, 37, 3.25, True),
    (BASE_I2V, '854c04a4c8a53d990b418c7478f0802c0fc8c726', 12, 1.0, True),
    (BASE_I2V, '854c04a4c8a53d990b418c7478f0802c0fc8c726', 12, 6.0, False),
    (I2V, 'a' * 40, 50, 6.0, False),
])
def test_i2v_guidance_is_owned_by_the_actual_immutable_loader(repo, revision, steps, guidance, allowed):
    token = issue_pipeline_instance_token(model_type=MODEL, repo_id=repo, repo_source='hub', revision=revision)
    state = _issue_workflow_state(token=token, pipeline_class=MODEL, workflow_id='image2video',
        completed_stage='image_encoder', state={'image_embeds': object()})
    pipeline, node = Pipeline(), WorkflowHunyuanVideo15Denoise()
    with patch.object(node, '_prepare_pipeline', return_value=(token, pipeline)), patch(
        'modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed', return_value='generator'):
        values = dict(pipeline_components={}, pipeline_class=MODEL, workflow_id='image2video',
            block_path='denoise', state_in=state, num_frames=121,
            num_inference_steps=steps, guidance_scale=guidance, seed=42)
        if not allowed:
            with pytest.raises(ValueError, match='fixed guidance scale|workflow/revision'):
                node.execute(**values)
            assert not pipeline.calls
            return
        node.execute(**values)
    assert pipeline.guider.guidance_scale == guidance
    assert pipeline.calls[0]['num_inference_steps'] == steps
    assert 'width' not in pipeline.calls[0] and 'height' not in pipeline.calls[0]
