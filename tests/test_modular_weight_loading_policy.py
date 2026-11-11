"""Pure native loading policy identity; no model/device imports."""

import ast
from pathlib import Path

import pytest

from modiff.modular_runtime_policy import normalize_modular_runtime_policy, runtime_policy_from_settings


SOURCE = Path(__file__).parents[1] / 'modules/ModularDiffusers/loaders.py'


def loader_function(name):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[name]


def test_native_descriptor_exposes_explicit_advanced_loading_mode_with_cpu_default():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    owner = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'ModelsLoader')
    params = next(item.value for item in owner.body if isinstance(item, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'params' for target in item.targets))
    field = next((value for key, value in zip(params.keys, params.values)
                  if isinstance(key, ast.Constant) and key.value == 'weight_loading_mode'), None)
    assert field is not None, 'An Auto-selected CUDA strategy must be an ordinary explicit loader control.'
    value = ast.literal_eval(field)
    assert value['type'] == 'string'
    assert value['default'] == 'cpu'
    assert value['options'] == ['cpu', 'cuda']
    assert value['fieldOptions']['controlTier'] == 'advanced'


def test_cpu_loading_keeps_historical_policy_identity_and_cuda_is_explicit():
    old = {'attention_backend': 'inherit', 'vae_slicing': None, 'vae_tiling': None}
    assert normalize_modular_runtime_policy(weight_loading_mode='cpu') == old
    assert normalize_modular_runtime_policy() == old
    assert normalize_modular_runtime_policy(weight_loading_mode='cuda') == {**old, 'weight_loading_mode': 'cuda'}
    assert runtime_policy_from_settings({'weight_loading_mode': 'cuda'}) == {**old, 'weight_loading_mode': 'cuda'}


@pytest.mark.parametrize('name', ['transformer', 'vae', 'text_encoder'])
def test_cuda_loading_tags_every_staged_heavy_component(name):
    policy = {'attention_backend': 'inherit', 'vae_slicing': None, 'vae_tiling': None,
              'weight_loading_mode': 'cuda'}
    assert loader_function('_component_runtime_policy')(name, policy)['weight_loading_mode'] == 'cuda'


@pytest.mark.parametrize('value', [None, '', 'CUDA', 'auto', True, 1, [], {}])
def test_loading_policy_rejects_undeclared_values(value):
    with pytest.raises(ValueError, match='loading mode'):
        normalize_modular_runtime_policy(weight_loading_mode=value)


@pytest.mark.parametrize('mode', ['cpu', 'cuda'])
def test_actual_consumed_loader_input_projection_records_loading_mode_separately_from_sparse_policy(mode):
    from modiff.execution_input_provenance import capture_generation_inputs

    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    owner = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'ModelsLoader')
    execute = next(item for item in owner.body if isinstance(item, ast.FunctionDef) and item.name == 'execute')
    projection = next(item for item in execute.body if isinstance(item, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'generation_inputs' for target in item.targets))
    namespace = {'model_type': 'ZImageModularPipeline', 'real_repo_id': 'Tongyi-MAI/Z-Image-Turbo',
                 'revision': 'a' * 40, 'dtype': 'bfloat16', 'device': 'cuda:0', 'auto_offload': True,
                 'offload_mode': 'group_disk', 'quant_config': None, 'weight_loading_mode': mode,
                 'runtime_policy': normalize_modular_runtime_policy(weight_loading_mode=mode)}
    exec(compile(ast.Module(body=[projection], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    receipt = capture_generation_inputs('models', {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader'},
                                       namespace['generation_inputs'])
    assert receipt['fields']['weight_loading_mode'] == {'value': mode, 'source': 'literal'}
    if mode == 'cpu':
        assert 'weight_loading_mode' not in namespace['runtime_policy']
