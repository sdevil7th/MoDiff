"""Actual registered publication and typed policy, CPU metadata proof only."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import modules
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.model_artifact_catalog import catalog_profile_repository_revisions
from modiff.server import WebServer

SUPER = 'cosmos3-super-text-to-image:official-modular-workflow'
REDUX = 'flux-redux:direct'


@pytest.fixture
def publication(monkeypatch, tmp_path):
    monkeypatch.setattr(WebServer, '_optimization_runtime_context', lambda _self, **_kwargs: (None, {}, {}))
    monkeypatch.setattr('modiff.server.public_optimization_catalog', lambda *_args, **_kwargs: {'capabilities': []})
    monkeypatch.setattr('modiff.NodeBase.NodeBase.__init__', lambda *_args, **_kwargs: pytest.fail('Node construction'))
    def publish(registry=modules.MODULE_MAP):
        server = WebServer(registry, work_dir=str(tmp_path), data_dir=str(tmp_path / 'data'))
        return server._build_model_capabilities_payload()
    return publish


@pytest.mark.parametrize('identity', [SUPER, REDUX])
def test_actual_registry_publishes_exact_private_declaration_and_identical_copies(publication, identity):
    payload = publication()
    expected = DIFFUSERS_EXECUTION_PROFILES[identity].adapter_authoring.to_public_dict()
    assert expected['requiredControls']
    copies = [row for rows in [payload['diffusersExecutionProfiles'], *[
        cap['executionProfiles'] for cap in payload['capabilities'] + payload['experimentalCapabilities']
    ]] for row in rows if row['id'] == identity]
    assert len(copies) >= 2
    selected = DIFFUSERS_EXECUTION_PROFILES[identity]
    assert selected.live_proof is False
    if identity == SUPER:
        assert selected.lora_loadable_components == ()
        assert selected.modes == ('text_to_image',)
        assert selected.supported_offload_modes == ('none',)
        for cap in payload['capabilities']:
            if any(row['id'] == identity for row in cap['executionProfiles']):
                assert cap['supportsLora'] is False and cap['autoEligible'] is False
                assert cap['liveProof'] is False and cap['galleryEligible'] is False
                assert cap['qualifiedModes'] == []
        assert copies[0]['optionalRuntimeRequirement']['profileIds'] == ['cosmos-guardrail-0.3.1']
        assert copies[0]['optionalRuntimeRequirement']['requiredNow'] is True
    else:
        assert selected.modes == ('edit_image', 'multi_image_reference_edit')
        assert selected.supported_offload_modes == ('model_cpu', 'sequential_cpu', 'group_cpu', 'group_disk')
        assert copies[0]['optionalRuntimeRequirement']['requiredNow'] is False
    for row in copies:
        assert row['adapterAuthoring'] == expected
        assert row['repository_revisions'] == expected['repositoryRevisions']
        assert row['live_proof'] is False
        assert 'lora_loadable_components' not in row


@pytest.mark.parametrize('identity,field,change', [
    (SUPER, 'quant_config', 'missing'), (SUPER, 'quant_config', 'not-dict'),
    (SUPER, 'quant_config', 'empty-types'), (SUPER, 'quant_config', 'none-type'),
    (SUPER, 'quant_config', 'malformed-type'), (SUPER, 'quant_config', 'empty-type-string'),
    (SUPER, 'quant_config', 'control-type'), (SUPER, 'quant_config', 'too-many-types'),
    (SUPER, 'quant_config', 'too-long-type'), (SUPER, 'quant_config', 'default'),
    (SUPER, 'quant_config', 'value'), (SUPER, 'quant_config', 'required'),
    (SUPER, 'quant_config', 'c1-type'), (SUPER, 'quant_config', 'unicode-line-type'),
    (SUPER, 'quant_config', 'unicode-paragraph-type'), (SUPER, 'quant_config', 'whitespace-type'), (SUPER, 'quant_config', 'required-string'),
    (SUPER, 'quant_config', 'required-number'), (SUPER, 'quant_config', 'output'), (SUPER, 'quant_config', 'not-input'),
    (SUPER, 'auto_offload', 'wrong-bool-type'), (SUPER, 'offload_mode', 'wrong-string-type'),
    (SUPER, 'offload_mode', 'wrong-options'), (SUPER, 'offload_mode', 'malformed-options'),
    (REDUX, 'prequantized_transformer', 'empty-types'),
    (REDUX, 'prequantized_transformer', 'default'),
    (REDUX, 'device_map', 'wrong-options'),
])
def test_actual_publisher_refuses_incompatible_registered_control_metadata(publication, identity, field, change):
    profile = DIFFUSERS_EXECUTION_PROFILES[identity]
    registry = dict(modules.MODULE_MAP)
    registry[profile.loader_module] = dict(registry[profile.loader_module])
    owner = deepcopy(registry[profile.loader_module][profile.loader_action])
    registry[profile.loader_module][profile.loader_action] = owner
    definition = owner['params'][field]
    if change == 'missing': owner['params'].pop(field)
    elif change == 'not-dict': owner['params'][field] = []
    elif change == 'empty-types': definition['type'] = []
    elif change == 'none-type': definition['type'] = None
    elif change == 'malformed-type': definition['type'] = [{}]
    elif change == 'empty-type-string': definition['type'] = ['']
    elif change == 'c1-type': definition['type'] = ['quant\x80config']
    elif change == 'unicode-line-type': definition['type'] = ['quant\u2028config']
    elif change == 'unicode-paragraph-type': definition['type'] = ['quant\u2029config']
    elif change == 'whitespace-type': definition['type'] = [' ']
    elif change == 'required-string': definition['required'] = 'true'
    elif change == 'required-number': definition['required'] = 1
    elif change == 'control-type': definition['type'] = ['quant_config\n']
    elif change == 'too-many-types': definition['type'] = ['quant_config'] * 17
    elif change == 'too-long-type': definition['type'] = ['x' * 129]
    elif change == 'default': definition['default'] = 'nonempty'
    elif change == 'value': definition['value'] = 'nonempty'
    elif change == 'required': definition['required'] = True
    elif change == 'output': definition['display'] = 'output'
    elif change == 'not-input': definition['display'] = 'text'
    elif change == 'wrong-bool-type': definition['type'] = 'string'
    elif change == 'wrong-string-type': definition['type'] = 'boolean'
    elif change == 'wrong-options': definition['options'] = ['group_disk']
    elif change == 'malformed-options': definition['options'] = 'none'
    else: raise AssertionError(change)
    with pytest.raises(ValueError, match='Adapter authoring control'):
        publication(registry)


@pytest.mark.parametrize('identity', [SUPER, REDUX])
@pytest.mark.parametrize('owner', [None, [], object()])
def test_actual_publisher_refuses_nonmetadata_registered_owner(publication, identity, owner):
    profile = DIFFUSERS_EXECUTION_PROFILES[identity]
    registry = dict(modules.MODULE_MAP)
    registry[profile.loader_module] = dict(registry[profile.loader_module])
    registry[profile.loader_module][profile.loader_action] = owner
    with pytest.raises(ValueError): publication(registry)


@pytest.mark.parametrize('identity', [SUPER, REDUX])
def test_profile_transport_pairing_remains_exact(identity):
    profile = DIFFUSERS_EXECUTION_PROFILES[identity]
    foreign = 'modules.DiffusersImage.LoadAdapter' if identity == SUPER else 'modules.ModularDiffusers.Lora'
    with pytest.raises(ValueError, match='incompatible adapter authoring policy'):
        replace(profile, adapter_authoring=replace(profile.adapter_authoring, adapter_node_key=foreign))


@pytest.mark.parametrize('controls', [[], (('prototype', 'none'),), (('offload_mode', 1),),
    (('offload_mode', float('nan')),), (('offload_mode', ''),), (('offload_mode', ' none'),),
    (('offload_mode', 'none\n'),), (('offload_mode', 'x' * 129),),
    (('offload_mode', 'none'), ('offload_mode', 'none')),
    tuple(('control_' + str(index), None) for index in range(33)),
    (('offload_mode', 'n\x80one'),), (('offload_mode', 'n\x9fone'),),
    (('offload_mode', 'n\u2028one'),), (('offload_mode', 'n\u2029one'),),
    (('offload_mode', '\U0001f600' * 65),),
])
def test_typed_policy_rejects_unbounded_or_ambiguous_scalar_controls(controls):
    with pytest.raises(ValueError):
        replace(DIFFUSERS_EXECUTION_PROFILES[SUPER].adapter_authoring, required_controls=controls)


def test_unknown_control_is_not_synthetic_permission():
    policy = replace(DIFFUSERS_EXECUTION_PROFILES[SUPER].adapter_authoring,
                     required_controls=(('unknown_control', None),))
    params = modules.MODULE_MAP['modules.ModularDiffusers']['ModelsLoader']['params']
    with pytest.raises(ValueError): policy.validate_required_controls(params)


def test_existing_six_key_policies_and_single_catalog_read_are_preserved(publication):
    from test_image_adapter_authoring_policy import FLUX_POLICIES, EDIT_POLICIES, FLUX3_PUBLIC_POLICIES
    from test_audio_adapter_authoring_policy import EXPECTED as ACE_POLICY, IDENTITY as ACE
    with patch('modiff.model_artifact_catalog.catalog_profile_repository_revisions',
               wraps=catalog_profile_repository_revisions) as catalogue:
        payload = publication()
    assert catalogue.call_count == 1
    profiles = {row['id']: row for row in payload['diffusersExecutionProfiles']}
    existing = {**FLUX_POLICIES, **EDIT_POLICIES}
    for identity, expected in existing.items():
        assert profiles[identity]['adapterAuthoring'] == expected
        assert set(profiles[identity]['adapterAuthoring']) == {
            'schemaVersion', 'mediaKind', 'adapterNodeKey', 'tasks', 'maxOwnedAdapters', 'repositoryRevisions'}
    assert profiles[ACE]['adapterAuthoring'] == ACE_POLICY
    assert profiles[ACE]['repository_revisions'] == ACE_POLICY['repositoryRevisions']
    for identity, expected in FLUX3_PUBLIC_POLICIES.items():
        assert profiles[identity]['adapterAuthoring'] == expected
        assert set(profiles[identity]['adapterAuthoring']) == {
            'schemaVersion', 'mediaKind', 'adapterNodeKey', 'tasks', 'maxOwnedAdapters',
            'repositoryRevisions', 'requiredControls'}
        assert profiles[identity]['repository_revisions'] == expected['repositoryRevisions']
        assert profiles[identity]['live_proof'] is False
    for identity, row in profiles.items():
        if identity not in {*existing, *FLUX3_PUBLIC_POLICIES, SUPER, REDUX, ACE}:
            assert 'adapterAuthoring' not in row
