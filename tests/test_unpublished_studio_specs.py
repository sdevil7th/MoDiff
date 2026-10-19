"""Future source declarations cannot break publication on the installed runtime."""
from copy import deepcopy
from pathlib import Path

import pytest

from modiff import studio_execution_specs as specs


@pytest.fixture
def private_specs(monkeypatch):
    selected = {key: deepcopy(value) for key, value in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.items()
                if value['profile'].get('public') is False}
    assert selected
    monkeypatch.setattr(specs, 'STUDIO_EXECUTION_SPEC_DEFINITIONS', selected)
    return selected


def test_default_publication_never_resolves_unpublished_runtime_classes(private_specs, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unpublished execution schemas must not be resolved by public discovery.')
    monkeypatch.setattr(specs, '_execution_spec_role_params', forbidden)
    # No registry exists for these future APIs on the ordinary reviewed wheel.
    assert specs.validate_studio_execution_specs({}) == []


def test_explicit_source_audit_still_enforces_the_actual_registry(private_specs):
    with pytest.raises(ValueError, match='unknown node'):
        specs.validate_studio_execution_specs({}, include_private=True)


def test_exact_private_lookup_remains_available_without_publication(private_specs):
    for definition in private_specs.values():
        selected = specs.studio_execution_spec_for_pair(definition['modelType'], definition['mode'])
        assert selected is not None
        assert selected['executionProfileId'] == definition['profile']['id']
        assert specs.studio_execution_spec_for_pair(definition['modelType'], definition['mode'],
                                                    content_hash=selected['contentHash']) == selected
    assert specs.validate_studio_execution_specs({}) == []


@pytest.mark.parametrize('value', ['true', 1, None, {}])
def test_source_audit_opt_in_is_not_a_truthy_transport_value(value):
    with pytest.raises(ValueError, match='explicit boolean'):
        specs.validate_studio_execution_specs({}, include_private=value)


def test_new_storage_field_only_enters_explicit_recipe_blocks():
    future = next(value for value in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
                  if value['profile'].get('public') is False)
    historical = next(value for value in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
                      if value['profile'].get('public', True)
                      and value['profile']['loader_module'] == 'modules.ModularDiffusers'
                      and value['profile']['loader_action'] == 'ModelsLoader'
                      and any(role[1] == 'modules.ModularDiffusers.ModelsLoader' for role in value.get('roles', ()))
                      and not any(binding[1] == 'weight_format' for binding in value.get('bindings', ())))
    field = {'type': 'string', 'default': 'native', 'hidden': True, 'options': ['native']}
    node = {'params': {'weight_format': field, 'repo_id': {'type': 'string'}}}
    def selected(definition):
        identity = next(key for key, value in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.items() if value is definition)
        return specs._execution_spec_role_params(specs._public_spec(identity, definition),
                                                 'modules.ModularDiffusers.ModelsLoader', node)
    assert 'weight_format' not in selected(historical)
    assert selected(future)['weight_format'] == field
    assert node['params']['weight_format'] == field


def test_unpublished_specs_never_grant_executable_source_coverage():
    from modiff import upstream_coverage as coverage

    _version, rows = coverage._pipeline_coverage(
        Path(__file__).resolve().parents[1], coverage.reviewed_diffusers_source())
    exported = {row['name'] for row in rows}
    expected = {
        (identifier, definition['modelType'], definition['mode'])
        for identifier, definition in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.items()
        if definition['profile'].get('public') is not False
        and definition['profile']['pipeline_class'] in exported
    }
    actual = {
        (item['id'], item['modelType'], item['mode'])
        for row in rows for item in row['exactExecutionSpecs']
    }
    assert actual == expected
    public_classes = {
        definition['profile']['pipeline_class']
        for definition in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
        if definition['profile'].get('public') is not False
    }
    private_only = {
        definition['profile']['pipeline_class']
        for definition in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
        if definition['profile'].get('public') is False
    } - public_classes
    assert private_only
    by_name = {row['name']: row for row in rows}
    for name in private_only:
        assert by_name[name]['status'] in {'research-blocked', 'contract-only', 'intentionally-excluded'}
        assert by_name[name]['exactExecutionSpecs'] == []


def test_unpublished_spec_cannot_qualify_an_equivalent_target(monkeypatch):
    from modiff import upstream_coverage as coverage

    private = next(definition for definition in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
                   if definition['profile'].get('public') is False)
    target = private['profile']['pipeline_class']
    monkeypatch.setattr(coverage, '_EQUIVALENT_PIPELINE_TARGETS', {'FluxPriorReduxPipeline': (target,)})
    with pytest.raises(coverage.UpstreamCoverageError, match='not executable'):
        coverage._pipeline_coverage(Path(__file__).resolve().parents[1], coverage.reviewed_diffusers_source())
