"""The planner and loader normalize one bounded policy before model imports."""
import subprocess
import sys

import pytest

from modiff.modular_runtime_policy import normalize_modular_runtime_policy, runtime_policy_from_settings


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
