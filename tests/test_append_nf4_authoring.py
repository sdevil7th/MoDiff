"""Explicit append precision choices; contract tests without model execution."""
from copy import deepcopy

import pytest

from modiff import ancillary_execution as ancillary
from modiff import studio_execution_specs as specs
from test_constrained_ancillary_authoring import _preview


IDENTITIES = tuple(
    f'{variant}:official:{mode}:prompt-preview-append-nf4:v1'
    for variant in ('flux3-klein', 'flux3-klein-base')
    for mode in ('text-to-image', 'edit-image'))


@pytest.mark.parametrize('identity', IDENTITIES)
def test_append_nf4_is_explicit_fixed_and_keeps_authored_layout(identity):
    import modules
    graph, parent = _preview(modules.MODULE_MAP, identity)
    child = parent['ancillaryOperations'][0]
    request = graph['nodes']['proposal']['params']
    request['prompt']['value'] = '  A paper <subject>.\n'
    before = deepcopy(graph)
    literals = {(role, name): value for role, name, value in child['initialValues']}
    assert literals['provider', 'dtype'] == 'bfloat16'
    assert literals['provider', 'quantization_mode'] == 'bnb_4bit'
    assert literals['provider', 'device'] == 'cuda:0'
    assert literals['provider', 'revision'] == '3f3edb52b2c55de0d49feab28598b68f7f8e4830'
    assert literals['proposal', 'max_input_tokens'] == 4096
    assert literals['proposal', 'operation'] == 'enhance_prompt_append_details'
    assert child['layoutPolicy'] == 'append-details-preserve-layout-v1'
    assert 'authoredLayoutConstraint' not in child
    assert ancillary.validate_execution_purpose(graph, modules.MODULE_MAP) == graph['runtimeHints']['executionPurpose']
    assert graph == before
    from modiff.prompt_enhancement_output import compose_append_details
    result = compose_append_details(request['prompt']['value'], request['layout_conditioning']['value'],
                                   'Soft morning light.', image_count=1 if 'edit-image' in identity else 0)
    assert result['caption'] == request['prompt']['value'] + '\n\nSoft morning light.'
    assert result['layoutConditioning'] == request['layout_conditioning']['value']


@pytest.mark.parametrize('identity', IDENTITIES)
@pytest.mark.parametrize('role,field,value', [
    ('provider', 'dtype', 'float32'), ('provider', 'quantization_mode', 'none'),
    ('provider', 'device', 'cpu'), ('provider', 'revision', 'a' * 40),
    ('proposal', 'operation', 'enhance_prompt'), ('proposal', 'max_input_tokens', 4095),
])
def test_append_nf4_has_no_precision_or_operation_fallback(identity, role, field, value):
    import modules
    graph, _parent = _preview(modules.MODULE_MAP, identity)
    graph['nodes']['proposal']['params']['prompt']['value'] = 'A <subject>.'
    graph['nodes'][role]['params'][field] = {'value': value}
    before = deepcopy(graph)
    with pytest.raises(ValueError):
        ancillary.validate_execution_purpose(graph, modules.MODULE_MAP)
    assert graph == before


def test_append_nf4_choice_keeps_fp32_default_and_all_existing_parent_receipts(monkeypatch):
    import modules
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    profiles = [DIFFUSERS_EXECUTION_PROFILES[f'{variant}:official-provider:v1'].to_public_dict()
                for variant in ('flux3-klein', 'flux3-klein-base')]
    for row in specs.studio_operation_authoring_selections(profiles):
        assert row['authoringRecipeId'].endswith(':prompt-preview-append-fp32:v1')
        matches = [choice for choice in row['layoutAuthoringRecipeChoices']
                   if choice['authoringRecipeId'] in IDENTITIES]
        assert len(matches) == 1 and 'BF16 + NF4' in matches[0]['label']
        assert matches[0]['authoringRecipeId'] not in {
            choice['authoringRecipeId'] for choice in row['rawAuthoringRecipeChoices']}
    previous = [*ancillary.PRIVATE_PARENTS,
                *(identity for identity in ancillary.PUBLIC_PARENTS if identity not in IDENTITIES)]
    assert len(previous) == 33
    before = {identity: ancillary.resolve_parent(modules.MODULE_MAP, identity) for identity in previous}
    with monkeypatch.context() as old:
        for identity in IDENTITIES:
            old.delitem(ancillary.PUBLIC_PARENTS, identity)
            old.delitem(specs._OPERATION_AUTHORING_RECIPES, identity)
        assert {identity: ancillary.resolve_parent(modules.MODULE_MAP, identity) for identity in previous} == before
