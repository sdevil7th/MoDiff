"""Ordinary exact current artifact selection is not a cross-variant override."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import test_flux3_native_lora as native_tests

runtime = native_tests.runtime


def test_actual_factory_self_variant_reaches_normal_selected_owner(runtime, tmp_path):
    captured = json.loads((Path(__file__).parent / 'fixtures/flux3-split-stages-client.v1.json').read_text())['apiGraph']
    bindings = captured['runtimeHints']['executionPurpose']['bindings']
    actual = captured['nodes'][bindings[0]['parentRoleNodes']['models']]['params']
    pair = native_tests.PAIRS[1]
    assert actual['repo_id']['value'] == {'source': 'hub', 'value': pair[1]}
    assert actual['revision']['value'] == pair[2]
    assert actual['reviewed_variant']['value'] == pair[1]
    source = native_tests.source_graph(pair, [native_tests.descriptor(tmp_path, original=True)])
    source['nodes']['models']['params']['reviewed_variant'] = deepcopy(actual['reviewed_variant'])
    node = runtime.run(source)
    spec = node.loader.get_component_spec('transformer')
    assert spec.pretrained_model_name_or_path == pair[1] and spec.revision == pair[2]
    assert node._native_disk_lora_state['componentComplete'] is True
    assert node.loader.transformer.active_adapters() == ['one']


@pytest.mark.parametrize('pair', native_tests.PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize('mutation', ['other_variant', 'wrong_revision', 'wrong_repository', 'mutable_revision'])
def test_self_variant_never_broadens_exact_artifact_authority(runtime, tmp_path, pair, mutation):
    source = native_tests.source_graph(pair, [native_tests.descriptor(tmp_path)])
    params = source['nodes']['models']['params']
    params['reviewed_variant'] = {'value': pair[1]}
    other = next(value for value in native_tests.PAIRS if value != pair)
    field, value = {'other_variant': ('reviewed_variant', other[1]), 'wrong_revision': ('revision', other[2]),
                    'wrong_repository': ('repo_id', {'source': 'hub', 'value': other[1]}),
                    'mutable_revision': ('revision', 'main')}[mutation]
    params[field] = {'value': value}
    with pytest.raises((ValueError, RuntimeError)):
        runtime.run(source)
    assert runtime.events == []
