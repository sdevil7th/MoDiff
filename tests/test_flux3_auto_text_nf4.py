"""Explicit parent NF4 is a separate graph-bound Auto recipe."""
from copy import deepcopy

import pytest

from modiff import flux3_auto_resource as resources
from modiff import flux3_text_encoder_quantization as quantization
from modiff import workflow_auto_resource as planner
from modiff.auto_resource import auto_resource_history_key
from test_flux3_auto_resource import PROFILES, actual_plan, flux_graph  # noqa: F401


def nf4_graph(profile, task='text_to_image'):
    graph = flux_graph(profile, task)
    owner = graph['nodes']['models']['params']
    owner['weight_format']['value'] = 'nvfp4'
    repository = owner['repo_id']['value']
    fields = {'model_id': repository, 'subfolder': 'text_encoder', 'component': 'text_encoder',
        'quant_type': 'bnb_4bit', 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': 'bfloat16',
        'bnb_4bit_use_double_quant': True, 'llm_int8_threshold': 6.0,
        'llm_int8_has_fp16_weight': False, 'llm_int8_skip_modules': [],
        'skip_module_names': 'model.visual,lm_head'}
    graph['nodes']['quantization'] = {'module': 'modules.ModularDiffusers', 'action': 'QuantizationConfigNode',
        'params': {name: {'value': value} for name, value in fields.items()}}
    owner['quant_config'] = {'sourceId': 'quantization', 'sourceKey': 'quantization_config'}
    graph['paths'][0].insert(0, 'quantization')
    return graph


@pytest.fixture
def nf4_plan(actual_plan, monkeypatch):  # noqa: F811
    plan, hardware, runtime = actual_plan
    hardware['accelerator']['capability'] = '8.9'
    monkeypatch.setattr(quantization, 'text_encoder_nf4_runtime_ready', lambda: True)

    def census(status, model, repository, revision, dtype):
        bound, reserve, dequant = resources.parameter_census(status, model, repository, revision, dtype)
        dequant['proofScope'] = 'largest_nvfp4_linear_dequantization_only'
        phases = {'proofScope': 'flux3_nvfp4_nf4_sequential_cuda_load_only',
            'systemRamBytes': 13 * 1024**3, 'vramBytes': 11 * 1024**3,
            'quantizedLinearCount': 252, 'requiresRuntimeHeadroomPolicy': True,
            'inferencePeakMeasured': False}
        return bound, reserve, dequant, phases

    monkeypatch.setattr(resources, 'nf4_parameter_census', census)
    return plan, hardware, runtime


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('references', (0, 1, 2))
def test_explicit_nf4_parent_task_uses_one_owner_with_full_phase_budget(nf4_plan, profile, references):
    from PIL import Image
    plan, hardware, _ = nf4_plan
    hardware['systemMemory']['availableBytes'] = 18 * 1024**3
    graph = nf4_graph(profile, 'edit_image' if references else 'text_to_image')
    before = deepcopy(graph)
    images = [Image.new('RGB', size) for size in [(512, 512), (1024, 1024)][:references]]
    result = plan(graph, images if references else None)
    assert result['canAutoRun'], result['issues']
    assert result['patches'] == [] and graph == before
    assert len(result['loaders']) == 1
    owner = result['loaders'][0]
    assert owner['settings']['textEncoderQuantization'] == quantization.inspect_text_encoder_nf4(graph['nodes'], 'models')
    assert owner['settings']['weightFormat'] == 'nvfp4'
    assert result['requirements']['systemRamBytes'] == 17 * 1024**3
    assert result['requirements']['vramBytes'] == 13 * 1024**3


@pytest.mark.parametrize('pool,available', [('systemMemory', 16 * 1024**3), ('accelerator', 12 * 1024**3),
    ('systemMemory', 0), ('accelerator', 0)])
def test_nf4_cold_phase_and_zero_memory_refuse_without_rewriting(nf4_plan, pool, available):
    plan, hardware, _ = nf4_plan
    hardware[pool]['availableBytes' if pool == 'systemMemory' else 'freeBytes'] = available
    graph = nf4_graph(PROFILES[0])
    before = deepcopy(graph)
    result = plan(graph)
    assert not result['canAutoRun'] and result['patches'] == [] and graph == before


@pytest.mark.parametrize('field,value', [('component', 'transformer'), ('bnb_4bit_quant_type', 'fp4'),
    ('bnb_4bit_compute_dtype', 'float32'), ('bnb_4bit_use_double_quant', False),
    ('skip_module_names', ''), ('llm_int8_skip_modules', [0]), ('subfolder', 'other')])
def test_unreviewed_quantization_controls_refuse_before_preparation(nf4_plan, field, value):
    plan, _, _ = nf4_plan
    graph = nf4_graph(PROFILES[0])
    graph['nodes']['quantization']['params'][field]['value'] = value
    result = plan(graph)
    assert not result['canAutoRun'] and not result['requiresPreparation']
    assert result['patches'] == []


def test_connected_custom_supplier_and_missing_combined_runtime_refuse(nf4_plan, monkeypatch):
    plan, _, _ = nf4_plan
    graph = nf4_graph(PROFILES[0])
    graph['nodes']['quantization']['module'] = 'custom.Untrusted'
    assert not plan(graph)['canAutoRun']
    graph = nf4_graph(PROFILES[0])
    graph['nodes']['connected'] = {'module': 'modules.Primitive', 'action': 'String',
        'params': {'value': {'value': 'text_encoder'}}}
    graph['paths'][0].insert(0, 'connected')
    graph['nodes']['quantization']['params']['component'] = {'sourceId': 'connected', 'sourceKey': 'value'}
    assert not plan(graph)['canAutoRun']
    monkeypatch.setattr(quantization, 'text_encoder_nf4_runtime_ready', lambda: False)
    assert not plan(nf4_graph(PROFILES[0]))['canAutoRun']


def test_nf4_configuration_enters_cache_and_history_identity():
    graph = nf4_graph(PROFILES[0])
    before = planner.workflow_owner_cache_key(graph, 'models')
    descriptor = quantization.inspect_text_encoder_nf4(graph['nodes'], 'models')
    candidate = {'weightFormat': 'nvfp4', 'textEncoderQuantization': descriptor}
    original = auto_resource_history_key(candidate)
    assert auto_resource_history_key({'weightFormat': 'nvfp4'}) != original
    graph['nodes']['quantization']['params']['bnb_4bit_use_double_quant']['value'] = False
    assert planner.workflow_owner_cache_key(graph, 'models') != before
    changed = deepcopy(candidate)
    changed['textEncoderQuantization']['signature'] = '0' * 64
    assert auto_resource_history_key(changed) != original


def test_loading_phase_record_cannot_be_absent_or_borrow_other_recipe():
    for candidate in ({}, {'boundedWeightFormatRecipe': resources.RECIPE_ID},
                      {'boundedWeightFormatRecipe': resources.NF4_RECIPE_ID,
                       'parameterLoadingPhases': {'systemRamBytes': 1, 'vramBytes': 1}}):
        with pytest.raises(ValueError, match='phase record'):
            resources.nf4_loading_requirements(candidate)


@pytest.mark.parametrize('capability', (None, '8.6', float('nan'), 'invalid'))
def test_nf4_requires_observed_supported_compute_capability(nf4_plan, capability):
    plan, hardware, _ = nf4_plan
    hardware['accelerator']['capability'] = capability
    result = plan(nf4_graph(PROFILES[0]))
    assert not result['canAutoRun'] and result['patches'] == []


def test_missing_explicit_declaration_cannot_leave_a_ready_proof(nf4_plan):
    from modiff.auto_resource import _bounded_fp8r_candidates, _auto_requirements_for_pair
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    _, hardware, _ = nf4_plan
    profile = DIFFUSERS_EXECUTION_PROFILES[PROFILES[0]]
    graph = nf4_graph(PROFILES[0])
    form = {'modelType': profile.model_type, 'mode': 'text_to_image', 'modelRepo': profile.default_repo,
        'executionProfileId': profile.id, 'weightFormat': 'nvfp4', 'dtype': 'bfloat16',
        'device': 'cuda:0', 'textEncoderQuantization': quantization.inspect_text_encoder_nf4(graph['nodes'], 'models')}
    requirements = deepcopy(_auto_requirements_for_pair(profile.model_type, 'text_to_image', profile.id))
    requirements.pop('explicitComponentQuantizationRecipe')
    candidate = _bounded_fp8r_candidates(form, hardware, requirements)[0]
    assert not candidate['canAutoRun'] and candidate['proof']['status'] == 'skipped'
