"""The planner and loader normalize one bounded policy before model imports."""
import subprocess
import sys

import pytest

from modiff.modular_runtime_policy import normalize_modular_runtime_policy, runtime_policy_from_settings


@pytest.mark.parametrize('model_type', ['MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline'])
@pytest.mark.parametrize('workflow', ['t2va', 'fl2va', 'ref2va'])
def test_only_reviewed_rocm_bfloat16_reference_routes_get_the_vision_policy(model_type, workflow):
    from modiff.modular_runtime_policy import ROCM_BF16_VISION_MATH, reviewed_reference_component_attention
    from modules.ModularDiffusers.loaders import _component_runtime_policy

    mapping = reviewed_reference_component_attention(model_type, workflow, rocm=True, bfloat16=True)
    assert mapping == ({'text_encoder': ROCM_BF16_VISION_MATH} if workflow == 'ref2va' else {})
    assert reviewed_reference_component_attention(model_type, workflow, rocm=False, bfloat16=True) == {}
    assert reviewed_reference_component_attention(model_type, workflow, rocm=True, bfloat16=False) == {}
    policy = {'component_vision_attention': mapping}
    assert _component_runtime_policy('text_encoder', policy) == (
        {'vision_attention': ROCM_BF16_VISION_MATH} if mapping else {})
    assert _component_runtime_policy('transformer', policy) == {}


def test_reference_attention_identity_cannot_reuse_or_mutate_another_tasks_encoder():
    from types import SimpleNamespace
    from modiff.modular_runtime_policy import ROCM_BF16_VISION_MATH
    from modules.ModularDiffusers.loaders import assert_explicit_component_runtime_policy

    encoder = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(_attn_implementation='eager')),
        _modiff_modular_runtime_policy={},
    )
    reference = {'component_vision_attention': {'text_encoder': ROCM_BF16_VISION_MATH}}
    with pytest.raises(ValueError, match='different attention/VAE policy'):
        assert_explicit_component_runtime_policy('text_encoder', encoder, reference)
    assert encoder.config.vision_config._attn_implementation == 'eager'
    encoder.config.vision_config._attn_implementation = ROCM_BF16_VISION_MATH
    encoder._modiff_modular_runtime_policy = {'vision_attention': ROCM_BF16_VISION_MATH}
    assert_explicit_component_runtime_policy('text_encoder', encoder, reference)
    with pytest.raises(ValueError, match='different attention/VAE policy'):
        assert_explicit_component_runtime_policy('text_encoder', encoder, {})


@pytest.mark.parametrize('settings,expected', [
    ({}, {'attention_backend': 'inherit', 'vae_slicing': None, 'vae_tiling': None}),
    ({'attention_backend': None, 'vae_slicing': False},
     {'attention_backend': 'inherit', 'vae_slicing': False, 'vae_tiling': None}),
    ({'attention_backend': '', 'inpaint_compatibility': 'native'},
     {'attention_backend': 'inherit', 'vae_slicing': None, 'vae_tiling': None}),
    ({'attention_backend': '_native_math', 'inpaint_compatibility': 'whole_v1'},
     {'attention_backend': '_native_math', 'vae_slicing': None, 'vae_tiling': None, 'inpaint_compatibility': 'whole_v1'}),
])
def test_policy_normalization_preserves_unset_false_and_compatibility(settings, expected):
    from modules.ModularDiffusers.loaders import modular_runtime_policy

    assert normalize_modular_runtime_policy(**settings) == expected
    assert modular_runtime_policy(**settings) == expected
    assert runtime_policy_from_settings({**settings, 'device': 'cpu', 'modelRepo': 'test/model'}) == expected


def test_native_policy_inspection_does_not_import_model_libraries():
    result = subprocess.run([sys.executable, '-c', '''
import sys
from modiff.workflow_auto_values import inspect_native_runtime_controls
nodes = {'load': {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader', 'params': {
    'attention_backend': {'value': None}, 'vae_slicing': {'value': False},
    'inpaint_compatibility': {'value': 'whole_v1'},
}}}
controls = inspect_native_runtime_controls(nodes, 'load', lambda nodes, node, field: nodes[node]['params'][field]['value'])
assert controls == {'attention_backend': 'inherit', 'vae_slicing': False, 'inpaint_compatibility': 'whole_v1'}
assert not {'torch', 'diffusers', 'transformers', 'peft'}.intersection(sys.modules)
'''], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
