"""Contract/ownership tests; no pretrained model, device or runtime installation."""
from copy import deepcopy
from types import SimpleNamespace
import weakref

import pytest

from modiff import ancillary_execution as ancillary
from modiff import ancillary_owner_lifecycle as lifecycle
from modiff import studio_execution_specs as specs

CONSTRAINED = tuple(
    f'{variant}:official:{mode}:prompt-preview-constrained-fp32:v1'
    for variant in ('flux3-klein', 'flux3-klein-base')
    for mode in ('text-to-image', 'edit-image'))
OPERATION = 'enhance_prompt_constrained_authored_v1'
POLICY = 'constrained-authored-table-v1'


def _preview(modules, identity):
    parent = ancillary.resolve_parent(modules, identity)
    child = parent['ancillaryOperations'][0]
    owners = {role: 'image-' + role for role, _key, _x, _y in parent['roles']}
    children = {'provider': 'provider', 'proposal': 'proposal'}
    nodes = {}
    for role, key, _x, _y in child['roles']:
        module, action = key.rsplit('.', 1)
        nodes[children[role]] = {'module': module, 'action': action, 'params': {}}
    for role, name, value in child['initialValues']:
        nodes[children[role]]['params'][name] = {'value': deepcopy(value)}
    nodes['proposal']['params'].update(
        prompt={'value': 'A paper crane.'},
        layout_conditioning={'value': {'schemaVersion': 2, 'coordinateSpace': 'normalized_xyxy', 'regions': [
            {'id': 'subject', 'description': ' paper crane ', 'action': 'place', 'targetBox': [.2, .2, .8, .8]}]}},
        model={'sourceId': 'provider', 'sourceKey': 'model'})
    reference = child['bindings']['owner']['referenceSource']
    if reference is not None:
        source = owners[reference['role']]
        nodes[source] = {'module': 'modules.Image', 'action': 'Load', 'params': {'file': {'value': ['first.png']}}}
        nodes['proposal']['params']['images'] = {'sourceId': source, 'sourceKey': reference['field']}
    binding = {'contractId': child['id'], 'contentHash': child['contentHash'],
        'parent': {**child['parent'], 'authoringContentHash': parent['contentHash']},
        'parentRoleNodes': owners, 'roleNodes': children}
    purpose = {'schemaVersion': 1, 'kind': 'explicit_preview', 'bindings': [binding],
        'target': {'nodeId': 'proposal', 'outputField': 'proposal_summary'}}
    # Reference loaders must precede the proposal that consumes their images.
    order = [node_id for node_id in nodes if node_id not in children.values()]
    order.extend((children['provider'], children['proposal']))
    return {'sid': None, 'nodes': nodes, 'paths': [order],
        'runtimeHints': {'resourceMode': 'expert', 'executionPurpose': purpose}}, parent


@pytest.mark.parametrize('identity', CONSTRAINED)
def test_four_distinct_public_parents_bind_exact_contract_and_fixed_provider(identity):
    import modules
    from modiff.prompt_enhancement_constraints import constraint_contract
    graph, parent = _preview(modules.MODULE_MAP, identity)
    child = parent['ancillaryOperations'][0]
    assert child['layoutPolicy'] == POLICY
    assert child['authoredLayoutConstraint'] == constraint_contract()
    assert child['parent']['executionProfileId'].endswith(':official-provider:v1')
    literals = {(role, name): value for role, name, value in child['initialValues']}
    assert literals['provider', 'dtype'] == 'float32'
    assert literals['provider', 'device'] == 'cuda:0'
    assert literals['provider', 'quantization_mode'] == 'none'
    assert literals['proposal', 'max_input_tokens'] == 4096
    assert literals['proposal', 'operation'] == OPERATION
    assert ancillary.validate_execution_purpose(graph, modules.MODULE_MAP) == graph['runtimeHints']['executionPurpose']
    append = ancillary.resolve_parent(modules.MODULE_MAP, identity.replace('constrained-fp32', 'append-fp32'))
    for key in ('baseSpecId', 'baseSpecContentHash', 'executionProfileIds', 'roles', 'edges', 'bindings', 'initialValues'):
        assert parent[key] == append[key]
    assert parent['contentHash'] != append['contentHash']
    assert child['contentHash'] != append['ancillaryOperations'][0]['contentHash']


@pytest.mark.parametrize('identity', CONSTRAINED)
@pytest.mark.parametrize('fault', ['null', 'empty', 'wrong-operation', 'nf4', 'reference'])
def test_constrained_recipe_refuses_unowned_geometry_or_different_controls(identity, fault):
    import modules
    graph, _parent = _preview(modules.MODULE_MAP, identity)
    params = graph['nodes']['proposal']['params']
    if fault == 'null': params['layout_conditioning']['value'] = None
    elif fault == 'empty': params['layout_conditioning']['value']['regions'] = []
    elif fault == 'wrong-operation': params['operation']['value'] = 'enhance_prompt_append_details'
    elif fault == 'nf4': graph['nodes']['provider']['params']['quantization_mode']['value'] = 'bnb_4bit'
    else:
        params['layout_conditioning']['value']['regions'][0]['source'] = {'referenceIndex': 10, 'box': [0, 0, 1, 1]}
    before = deepcopy(graph)
    with pytest.raises(ValueError): ancillary.validate_execution_purpose(graph, modules.MODULE_MAP)
    assert graph == before


def test_new_choices_are_explicit_bounded_disjoint_and_append_remains_default(monkeypatch):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    profiles = [DIFFUSERS_EXECUTION_PROFILES[identity].to_public_dict() for identity in (
        'flux3-klein:official-provider:v1', 'flux3-klein-base:official-provider:v1')]
    rows = specs.studio_operation_authoring_selections(profiles)
    assert len(rows) == 4
    for row in rows:
        choices = row['layoutAuthoringRecipeChoices']
        assert choices[0]['authoringRecipeId'] == row['authoringRecipeId']
        assert row['authoringRecipeId'].endswith(':prompt-preview-append-fp32:v1')
        assert choices[1]['authoringRecipeId'] in CONSTRAINED
        assert not {choice['authoringRecipeId'] for choice in choices} & {
            choice['authoringRecipeId'] for choice in row['rawAuthoringRecipeChoices']}
    first = rows[0]['authoringRecipeId']
    original = specs._NORMAL_OPERATION_LAYOUT_AUTHORING_CHOICES[first]
    for malformed in ((original[0], original[0]), tuple([original[0]] * 5), (original[1],),
                      (original[0], (rows[0]['rawAuthoringRecipeId'], 'wrong raw choice'))):
        with monkeypatch.context() as changed:
            changed.setitem(specs._NORMAL_OPERATION_LAYOUT_AUTHORING_CHOICES, first, malformed)
            with pytest.raises(ValueError): specs.studio_operation_authoring_selections(profiles)


def test_existing_seventeen_private_and_twelve_public_parent_dtos_do_not_depend_on_new_aliases(monkeypatch):
    import modules
    identities = [*ancillary.PRIVATE_PARENTS, *(identity for identity in ancillary.PUBLIC_PARENTS
                 if identity not in CONSTRAINED and ':prompt-preview-append-nf4:' not in identity)]
    assert len(identities) == 29
    actual = {identity: ancillary.resolve_parent(modules.MODULE_MAP, identity) for identity in identities}
    with monkeypatch.context() as old:
        for identity in CONSTRAINED:
            old.delitem(ancillary.PUBLIC_PARENTS, identity)
            old.delitem(specs._OPERATION_AUTHORING_RECIPES, identity)
        assert {identity: ancillary.resolve_parent(modules.MODULE_MAP, identity) for identity in identities} == actual
    assert all('authoredLayoutConstraint' not in parent['ancillaryOperations'][0] for parent in actual.values())


class _Processor:
    pass


def _scope(monkeypatch):
    scope = object.__new__(lifecycle._Scope)
    processor = _Processor()
    scope.preview, scope.auto = True, False
    scope.proposal, scope.provider, scope.graph_hash = 'proposal', 'provider', 'current-graph'
    scope.graph = {'nodes': {'proposal': {'params': {'operation': {'value': OPERATION}}}}}
    scope.server = SimpleNamespace(current_task={'current_node': 'proposal'})
    scope.preallocation_witness = (weakref.ref(processor), {'input': 'fingerprint'}, {'bound': 4096}, scope.graph_hash)
    scope._authored_constraint_deadline = 2800.0
    monkeypatch.setattr(lifecycle._Scope, 'check', lambda self: None)
    import time
    monkeypatch.setattr(time, 'monotonic', lambda: 1000.0)
    return scope, processor, SimpleNamespace(node_id='proposal')


def test_getter_carries_same_finite_deadline_without_reset_and_rechecks_processor_witness(monkeypatch):
    scope, processor, node = _scope(monkeypatch)
    token = lifecycle._SCOPE.set(scope)
    try:
        assert lifecycle.constrained_preallocation_context(node, processor) == 2800.0
        assert lifecycle.constrained_preallocation_context(node, processor) == 2800.0
        assert scope._authored_constraint_deadline == 2800.0
        with pytest.raises(ValueError, match='processed request changed'):
            lifecycle.constrained_preallocation_context(node, _Processor())
    finally: lifecycle._SCOPE.reset(token)


@pytest.mark.parametrize('fault', ['no-scope', 'auto', 'ordinary', 'no-witness', 'graph', 'node', 'expired', 'infinite', 'reset', 'bool'])
def test_getter_has_no_graph_deadline_or_implicit_scope_fallback(monkeypatch, fault):
    scope, processor, node = _scope(monkeypatch)
    if fault == 'auto': scope.auto = True
    elif fault == 'ordinary': scope.graph['nodes']['proposal']['params']['operation']['value'] = 'enhance_prompt'
    elif fault == 'no-witness': scope.preallocation_witness = None
    elif fault == 'graph': scope.graph_hash = 'changed'
    elif fault == 'node': node.node_id = 'other'
    elif fault == 'expired': scope._authored_constraint_deadline = 1000.0
    elif fault == 'infinite': scope._authored_constraint_deadline = float('inf')
    elif fault == 'reset': scope._authored_constraint_deadline = 2801.0
    elif fault == 'bool': scope._authored_constraint_deadline = True
    token = lifecycle._SCOPE.set(None if fault == 'no-scope' else scope)
    try:
        with pytest.raises(ValueError): lifecycle.constrained_preallocation_context(node, processor)
    finally: lifecycle._SCOPE.reset(token)


def test_explicit_constraint_reserves_additional_cpu_preparation_before_provider(monkeypatch):
    from modiff import workflow_auto_lifecycle
    from modiff.transformers_image_text_resources import image_text_cpu_preparation_requirements
    scope, _processor, _node = _scope(monkeypatch)
    captured = []
    class ReachedCapacity(Exception): pass
    def capacity(server, value):
        captured.append(value)
        raise ReachedCapacity
    monkeypatch.setattr(workflow_auto_lifecycle, 'assert_next_owner_capacity', capacity)
    with pytest.raises(ReachedCapacity): scope.before_node('provider')
    assert captured[0]['requirements']['systemRamBytes'] == image_text_cpu_preparation_requirements()['systemRamBytes'] + 256 * 1024**2
    assert captured[0]['device'] == 'cpu'
    assert captured[0]['parameterResidencyRequirements'] == {'systemRamBytes': 0, 'vramBytes': 0}
