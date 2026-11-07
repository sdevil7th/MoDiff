"""Actual declared Redux stage roles and immutable additional artifact."""
import pytest
from redux_stage_test_support import (PRIOR_REPO, BASE_REPO, BASE_REVISION)

@pytest.mark.parametrize('task', ['edit_image', 'multi_image_reference_edit'])
def test_current_redux_starter_requires_real_generic_encode_denoise_decode(task):
    from modiff.studio_execution_specs import studio_capability_definition, studio_execution_spec_for_pair
    spec = studio_execution_spec_for_pair('FluxReduxPipeline', task)
    assert spec['executionProfileId'] == 'flux-redux:direct'
    assert spec['defaultRepo'] == PRIOR_REPO
    capability = studio_capability_definition('FluxReduxPipeline')
    requirement, = capability['additionalRequirements']
    assert requirement['repo'] == BASE_REPO
    assert requirement['revision'] == BASE_REVISION
    keys = {role[1] for role in spec['roles']}
    assert {'modules.DiffusersImage.EncodeInputs',
            'modules.DiffusersImage.Denoise',
            'modules.DiffusersImage.DecodeLatents'} <= keys
    assert 'modules.DiffusersImage.Edit' not in keys
