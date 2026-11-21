"""Exact missing template stage contracts; no models or downloads."""
from dataclasses import asdict

import pytest

from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.studio_execution_specs import studio_execution_spec_for_pair
from modiff.video_stage_contract import declared_backend_video_stages, video_stage_spec


@pytest.mark.parametrize(('model_type', 'task'), [
    ('WanVideoPipeline', 'text_to_video'),
    ('WanVideoPipeline', 'video_to_video'),
    ('WanVideoPipeline', 'video_color_edit'),
    ('WanVACEPipeline', 'video_inpaint'),
    ('WanVACEPipeline', 'video_outpaint'),
    ('WanVACEPipeline', 'control_to_video'),
    ('WanImageToVideoPipeline', 'image_to_video'),
])
def test_agreed_fresh_templates_declare_real_separate_video_encoding(model_type, task):
    original = studio_execution_spec_for_pair(model_type, task)
    assert original is not None
    profile = DIFFUSERS_EXECUTION_PROFILES[original['executionProfileId']]
    artifact = {'repository': profile.default_repo,
                'revision': require_catalog_revision(profile.default_repo)}
    assert declared_backend_video_stages(asdict(profile), task, artifact, original), (
        f'{model_type}/{task} has no exact backend-owned video-stage admission')
    staged = video_stage_spec(original)
    encoders = [role for role, key, *_ in staged['roles']
                if key == 'modules.DiffusersVideo.EncodeInputs']
    assert len(encoders) == 1
    assert any(source == encoders[0] and output == 'encoded_inputs'
               for source, output, _target, _field in staged['edges'])
    # The migration must not lose any authored creator control merely because
    # it moves from the integrated operation into its real consuming stage.
    ordinary_fields = {field for role, field, _ in original['bindings']
                       if role == 'wanGenerate' and field != 'mode'}
    if model_type == 'WanVACEPipeline':
        ordinary_fields.remove('frame_rate')
        assert ('videoExport', 'fps', 'fps') in staged['bindings']
    actual_fields = {field for _role, field, _ in staged['bindings']}
    assert ordinary_fields <= actual_fields
