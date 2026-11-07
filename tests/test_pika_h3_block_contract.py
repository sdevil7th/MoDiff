"""The newer public hierarchy keeps its own exact source and no execution claim."""
from copy import deepcopy
import hashlib
import json

import pytest

from modiff import supplemental_modular_contracts as supplemental
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION


def rehash(document):
    body = {key: value for key, value in document.items() if key != 'contentHash'}
    document['contentHash'] = 'sha256:' + hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def test_pika_block_companion_keeps_exact_source_and_detached_structural_data():
    document = supplemental.reviewed_supplemental_modular_blocks(supplemental.PIKA_PIPELINE)
    assert document['source']['revision'] == supplemental.PIKA_SOURCE_REVISION
    assert document['blocks']['diffusersRevision'] == supplemental.PIKA_SOURCE_REVISION
    assert PINNED_DIFFUSERS_REVISION == '7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941'
    assert len(document['blocks']['blockDefinitions']) == 26
    assert {row['workflowId'] for row in document['blocks']['workflows']} == {'t2va', 'fl2va', 'ref2va'}
    assert document['scope'] == 'Source-generated structural companion only; no execution or hardware qualification.'
    document['blocks']['blockDefinitions'].clear()
    assert len(supplemental.reviewed_supplemental_modular_blocks(supplemental.PIKA_PIPELINE)['blocks']['blockDefinitions']) == 26
    assert supplemental.reviewed_supplemental_modular_blocks('UnknownPipeline') is None


@pytest.mark.parametrize('mutation', ['source', 'revision', 'workflow', 'path', 'unknown', 'boolean', 'hash'])
def test_pika_block_companion_rejects_substituted_sources_and_topology_even_with_a_new_outer_hash(mutation):
    document = deepcopy(supplemental.reviewed_supplemental_modular_blocks(supplemental.PIKA_PIPELINE))
    if mutation == 'source':
        document['source']['revision'] = 'main'
    elif mutation == 'revision':
        document['blocks']['diffusersRevision'] = PINNED_DIFFUSERS_REVISION
    elif mutation == 'workflow':
        document['blocks']['workflows'].pop()
    elif mutation == 'path':
        document['blocks']['workflows'][0]['placements'][0]['path'] = ['invented']
    elif mutation == 'unknown':
        document['execute'] = True
    elif mutation == 'boolean':
        document['schemaVersion'] = True
    rehash(document)
    if mutation == 'hash':
        document['contentHash'] = 'sha256:' + '0' * 64
    with pytest.raises(ValueError):
        supplemental.validate_supplemental_modular_blocks(document)


def test_exact_installed_provider_reproduces_the_block_companion_without_weights():
    from modiff.runtime_overlays import active_diffusers_provider
    provider = active_diffusers_provider()
    if not provider or provider.get('profileId') != supplemental.PIKA_RUNTIME_PROFILE:
        pytest.skip('Requires the exact installed reviewed video provider')
    assert supplemental.reproduce_installed_pika_blocks() == supplemental.reviewed_supplemental_modular_blocks(supplemental.PIKA_PIPELINE)
