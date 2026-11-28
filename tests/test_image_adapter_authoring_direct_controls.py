"""Pure control filtering using the actual private producer and recipe inspector.

These controls do not qualify an accepted recipe, resource fit or execution.
The actual direct populated-adapter resource accounting is reviewed separately.
"""
from copy import deepcopy

import pytest
from modiff import workflow_auto_resource as planner
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.workflow_auto_values import filter_adapter_authoring_candidates, inspect_execution_recipe_controls


def source(*, recipe=False):
    values = dict(offload_mode='none', auto_offload=False, device_map='none', quantization_mode='none')
    owner = {'module': 'modules.DiffusersImage', 'action': 'LoadPipeline',
             'params': {key: {'value': value} for key, value in values.items()}}
    owner['params']['prequantized_transformer'] = {'display': 'input'}
    nodes = {'owner': owner, 'adapter': {'module': 'modules.DiffusersImage', 'action': 'LoadAdapter',
             'params': {'pipeline': {'sourceId': 'owner', 'sourceKey': 'pipeline'},
                        'adapter_path': {'value': 'unit/populated-adapter'}}}}
    if recipe:
        owner['params']['execution_recipe'] = {'sourceId': 'recipe', 'sourceKey': 'execution_recipe'}
        nodes['recipe'] = {'module': 'modules.DiffusersRuntime', 'action': 'DiffusersExecutionRecipe',
                           'params': {'offload_mode': {'value': 'none'}, 'device_map': {'value': 'none'}}}
    return nodes


def filter_candidates(nodes, *, consumers=('adapter',), candidates=None, repository=None):
    profile = DIFFUSERS_EXECUTION_PROFILES['flux-redux:direct']
    assert profile.adapter_authoring.required_controls
    values, _bindings = inspect_execution_recipe_controls(nodes, 'owner', planner._literal)
    candidates = candidates if candidates is not None else [{
        'id': 'unit-none', 'artifactRevision': profile.adapter_authoring.repository_revisions[0][1],
        'offloadMode': 'none', 'autoOffload': False, 'quantizationMode': 'none',
    }]
    return filter_adapter_authoring_candidates(candidates, nodes=nodes, loader_id='owner',
        consumers=consumers, profile=profile, mode='edit_image',
        repository=repository or profile.default_repo, literal=planner._literal,
        recipe_controls=values, candidate_fields=planner._SETTINGS)


@pytest.mark.parametrize('recipe', [False, True])
def test_exact_none_declared_optional_null_controls_are_retained(recipe):
    nodes = source(recipe=recipe)
    before = deepcopy(nodes)
    assert filter_candidates(nodes)[0]['id'] == 'unit-none'
    assert nodes == before


def test_pure_recipe_placement_overrides_raw_offload_and_derives_auto():
    nodes = source(recipe=True)
    nodes['owner']['params']['offload_mode']['value'] = 'group_cpu'
    nodes['owner']['params']['auto_offload']['value'] = True
    assert filter_candidates(nodes)[0]['id'] == 'unit-none'


@pytest.mark.parametrize('raw,recipe', [('balanced', 'none'), ('cuda', 'none'), ('none', 'balanced')])
def test_raw_device_map_precedence_and_recipe_map_are_not_erased(raw, recipe):
    nodes = source(recipe=True)
    nodes['owner']['params']['device_map']['value'] = raw
    nodes['recipe']['params']['device_map']['value'] = recipe
    with pytest.raises(ValueError): filter_candidates(nodes)


@pytest.mark.parametrize('field,value', [('quantization_mode', 'bnb_4bit'),
                                       ('prequantized_transformer', {}), ('auto_offload', 0)])
def test_nonmatching_or_wrong_scalar_type_is_not_permission(field, value):
    nodes = source()
    nodes['owner']['params'][field] = {'value': value}
    # The candidate leaves this authored field unchanged in this control.
    candidate = {'id': 'unit-none', 'artifactRevision': 'c95859fbf7703ca4d6824b4da4407d7cd0434f81',
                 'offloadMode': 'none'}
    with pytest.raises(ValueError): filter_candidates(nodes, candidates=[candidate])


@pytest.mark.parametrize('change', ['missing', 'connected', 'output', 'inline-recipe', 'foreign-recipe', 'connected-recipe'])
def test_unknown_or_ambiguous_controls_cannot_establish_selected_permission(change):
    nodes = source(recipe=change in ('foreign-recipe', 'connected-recipe'))
    if change == 'missing': nodes['owner']['params'].pop('device_map')
    elif change == 'connected':
        nodes['owner']['params']['device_map'] = {'sourceId': 'custom', 'sourceKey': 'value'}
        nodes['custom'] = {'module': 'custom.User', 'action': 'Run', 'params': {}}
    elif change == 'output': nodes['owner']['params']['device_map']['display'] = 'output'
    elif change == 'inline-recipe': nodes['owner']['params']['execution_recipe'] = {'value': {}}
    elif change == 'foreign-recipe': nodes['recipe']['module'] = 'custom.User'
    else:
        nodes['recipe']['params']['offload_mode'] = {'sourceId': 'custom', 'sourceKey': 'value'}
        nodes['custom'] = {'module': 'custom.User', 'action': 'Run', 'params': {}}
    with pytest.raises(ValueError): filter_candidates(nodes)


@pytest.mark.parametrize('bad', ['revision', 'repository', 'unsafe-placement'])
def test_candidate_requires_the_current_immutable_pair_and_controls(bad):
    candidate = {'id': 'unit-none', 'artifactRevision': 'c95859fbf7703ca4d6824b4da4407d7cd0434f81',
                 'offloadMode': 'none', 'autoOffload': False}
    repository = None
    if bad == 'revision': candidate['artifactRevision'] = '0' * 40
    elif bad == 'repository': repository = 'black-forest-labs/FLUX.1-dev'
    else: candidate.update(offloadMode='group_cpu', autoOffload=True)
    with pytest.raises(ValueError): filter_candidates(source(), candidates=[candidate], repository=repository)


def test_vanilla_does_not_acquire_conditional_inspection_or_alter_candidates():
    nodes = source()
    nodes['owner']['params'].pop('device_map')
    candidates = [{'id': 'legacy-unsafe-for-adapter', 'offloadMode': 'group_cpu'}]
    assert filter_candidates(nodes, consumers=(), candidates=candidates) is candidates


def test_current_redux_declared_modes_cannot_be_promoted_into_none():
    profile = DIFFUSERS_EXECUTION_PROFILES['flux-redux:direct']
    assert 'none' not in profile.supported_offload_modes
    candidates = [{'id': mode, 'offloadMode': mode, 'autoOffload': True,
                   'artifactRevision': profile.adapter_authoring.repository_revisions[0][1]}
                  for mode in profile.supported_offload_modes]
    before = deepcopy(candidates)
    with pytest.raises(ValueError): filter_candidates(source(), candidates=candidates)
    assert candidates == before


@pytest.mark.parametrize('selection', [None, '', '  ', {'source': 'hub', 'value': ''}])
def test_empty_direct_adapter_chain_preserves_vanilla_placement(selection):
    nodes = source()
    nodes['adapter']['params']['adapter_path'] = {'value': selection}
    nodes['owner']['params']['offload_mode']['value'] = 'group_cpu'
    nodes['owner']['params']['auto_offload']['value'] = True
    candidate = {'id': 'ordinary-vanilla', 'offloadMode': 'group_cpu', 'autoOffload': True}
    candidates = [candidate]
    assert filter_candidates(nodes, candidates=candidates) is candidates


@pytest.mark.parametrize('selection', [False, {}, {'source': 'unknown', 'value': ''}])
def test_ambiguous_selection_does_not_take_the_empty_chain_bypass(selection):
    nodes = source()
    nodes['adapter']['params']['adapter_path'] = {'value': selection}
    with pytest.raises(ValueError):
        filter_candidates(nodes, candidates=[{'id': 'unsafe', 'offloadMode': 'group_cpu', 'autoOffload': True}])
