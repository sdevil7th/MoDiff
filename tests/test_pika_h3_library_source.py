"""Adding an approved catalog source preserves every existing definition."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from modiff.huggingface_node_library import build_huggingface_node_library, validate_huggingface_node_library
from modiff.supplemental_modular_contracts import PIKA_PIPELINE, PIKA_SOURCE_REVISION


@pytest.fixture(scope='module')
def library():
    return build_huggingface_node_library()


def test_mixed_library_keeps_v6_members_and_exact_existing_hashes(library):
    baseline = json.loads((Path(__file__).parent / 'fixtures/huggingface-library-v6-definition-hashes.v1.json').read_bytes())
    assert library['schemaVersion'] == 7
    assert library['diffusersRevision'] == baseline['catalogRevision']
    definitions = {row['id']: row for row in library['definitions']}
    for identity, digest in baseline['definitions'].items():
        assert definitions[identity]['schemaVersion'] == 6
        assert definitions[identity]['contentHash'] == digest, identity
    new = [row for row in definitions.values() if row['id'] not in baseline['definitions']]
    assert len(new) == 3
    assert {row['workflowId'] for row in new} == {'t2va', 'fl2va', 'ref2va'}
    assert all(row['pipelineClass'] == PIKA_PIPELINE and row['libraryRevision'] == PIKA_SOURCE_REVISION for row in new)
    assert len(library['sourceSupplements']) == 1
    assert set(library['sourceSupplements'][0]['definitionIds']) == {row['id'] for row in new}
    assert validate_huggingface_node_library(library) == library


def test_explicit_legacy_collection_is_still_valid_without_any_source_rewrite():
    legacy = build_huggingface_node_library(include_source_supplements=False)
    assert legacy['schemaVersion'] == 6 and 'sourceSupplements' not in legacy
    assert len(legacy['definitions']) == 139
    assert validate_huggingface_node_library(legacy) == legacy


@pytest.mark.parametrize('mutation', ['revision', 'runtime', 'membership', 'extra-source', 'definition-source', 'legacy-envelope'])
def test_new_source_declarations_cannot_substitute_provenance_or_membership(library, mutation):
    value = deepcopy(library)
    source = value['sourceSupplements'][0]
    if mutation == 'revision':
        source['libraryRevision'] = value['diffusersRevision']
    elif mutation == 'runtime':
        source['runtimeProfileId'] = 'unreviewed-runtime'
    elif mutation == 'membership':
        source['definitionIds'].pop()
    elif mutation == 'extra-source':
        value['sourceSupplements'].append(deepcopy(source))
    elif mutation == 'definition-source':
        next(row for row in value['definitions'] if row['pipelineClass'] == PIKA_PIPELINE)['libraryRevision'] = value['diffusersRevision']
    else:
        value['schemaVersion'] = 6
    with pytest.raises(ValueError):
        validate_huggingface_node_library(value)
