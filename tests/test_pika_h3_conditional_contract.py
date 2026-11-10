"""New source trees never relabel the preserved historical hierarchy."""
from copy import deepcopy

import pytest

from modiff.modular_conditional_contracts import (
    reviewed_modular_conditional_collection, reviewed_modular_conditional_snapshot,
    validate_modular_conditional_snapshot,
)
from modiff import supplemental_modular_contracts as source


def test_pika_original_tree_and_resolved_companions_are_source_bound_and_detached():
    document = source.reviewed_supplemental_modular_conditionals(source.PIKA_PIPELINE)
    snapshot = document['snapshot']
    assert snapshot['diffusersRevision'] == source.PIKA_SOURCE_REVISION
    assert len(snapshot['blockDefinitions']) == 32
    pipeline, = snapshot['pipelines']
    assert pipeline['pipelineClass'] == source.PIKA_PIPELINE
    assert len(pipeline['placements']) == 43 and len(pipeline['conditionals']) == 4
    assert {row['id'] for row in pipeline['workflows']} == set(source.PIKA_WORKFLOW_TASKS)
    with pytest.raises(ValueError, match='pinned Diffusers revision'):
        validate_modular_conditional_snapshot(snapshot)
    historical = reviewed_modular_conditional_snapshot()
    collection = reviewed_modular_conditional_collection()
    assert collection['schemaVersion'] == 2
    assert {key: value for key, value in {**collection, 'schemaVersion': 1}.items()
        if key != 'sourceSupplements'} == historical
    assert [s for s in collection['sourceSupplements'] if s['diffusersRevision'] == snapshot['diffusersRevision']] == [snapshot]
    collection['sourceSupplements'].clear()
    document['snapshot']['blockDefinitions'].clear()
    assert len(reviewed_modular_conditional_collection()['sourceSupplements'][0]['blockDefinitions']) == 32
    assert source.reviewed_supplemental_modular_conditionals('UnknownPipeline') is None


@pytest.mark.parametrize('mutation', ['source', 'revision', 'blocks', 'workflows', 'path', 'unknown', 'boolean'])
def test_pika_original_tree_rejects_forged_source_or_topology_even_with_outer_hash(mutation):
    document = deepcopy(source.reviewed_supplemental_modular_conditionals(source.PIKA_PIPELINE))
    if mutation == 'source':
        document['source']['revision'] = 'main'
    elif mutation == 'revision':
        document['snapshot']['diffusersRevision'] = reviewed_modular_conditional_snapshot()['diffusersRevision']
    elif mutation == 'blocks':
        document['blockContractHash'] = 'sha256:' + '0' * 64
    elif mutation == 'workflows':
        document['snapshot']['pipelines'][0]['workflows'].pop()
    elif mutation == 'path':
        document['snapshot']['pipelines'][0]['placements'][0]['path'] = ['invented']
    elif mutation == 'unknown':
        document['execute'] = True
    else:
        document['schemaVersion'] = True
    pipeline = document['snapshot']['pipelines'][0]
    pipeline['contentHash'] = source._contract_hash({key: value for key, value in pipeline.items() if key != 'contentHash'})
    document['contentHash'] = source._contract_hash({key: value for key, value in document.items() if key != 'contentHash'})
    with pytest.raises(ValueError):
        source.validate_supplemental_modular_conditionals(document)


def test_exact_installed_provider_reproduces_the_original_tree_without_weights():
    from modiff.runtime_overlays import active_diffusers_provider
    provider = active_diffusers_provider()
    if not provider or provider.get('profileId') != source.PIKA_RUNTIME_PROFILE:
        pytest.skip('Requires the exact installed reviewed video provider')
    assert source.reproduce_installed_pika_conditionals() == source.reviewed_supplemental_modular_conditionals(source.PIKA_PIPELINE)


@pytest.mark.parametrize('revision', ['main', True, 'd0b9b1b'])
def test_conditional_validator_never_accepts_a_mutable_expected_source(revision):
    with pytest.raises(ValueError, match='immutable revision'):
        validate_modular_conditional_snapshot({}, source_revision=revision)
