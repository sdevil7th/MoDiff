"""Pinned file selection is independent from hardware or model qualification."""
from copy import deepcopy
import json

import pytest

from modiff.joint_av_artifacts import (
    PATH, reviewed_joint_av_artifact_closure, validate_joint_av_artifact_closures,
)


H3 = 'MiniMaxH3ModularPipeline'
PIKA = 'PikaH3FlexModularPipeline'


@pytest.mark.parametrize('pipeline', [H3, PIKA])
@pytest.mark.parametrize('workflow', ['t2va', 'fl2va', 'ref2va'])
def test_finite_selection_contains_only_its_task_partition_and_required_notices(pipeline, workflow):
    task = reviewed_joint_av_artifact_closure(pipeline, workflow)
    names = {item['path'] for item in task['files']}
    assert task['transformerPartition'] == ('transformer_ref' if workflow == 'ref2va' else 'transformer')
    assert task['selectedByteSize'] == sum(item['byteSize'] for item in task['files'])
    assert len(task['files']) == (61 if pipeline == H3 else 35)
    if pipeline == H3:
        assert task['requiredAdapter'] is None
        assert task['selectedByteSize'] == 144051157679
        historical = json.loads((PATH.parent / 'minimax-h3-artifact-review.json').read_bytes())
        expected = {file['path']: file for key in ('shared', task['transformerPartition'])
                    for file in historical['artifactSelection']['partitions'][key]['weightFiles']}
        weights = {file['path']: file for file in task['files'] if file['path'].endswith('.safetensors')}
        assert weights.keys() == expected.keys()
        for name, file in weights.items():
            assert (file['sha256'], file['byteSize']) == (expected[name]['sha256'], expected[name]['byteSize'])
    else:
        assert {'NOTICE', 'licenses/MiniMax-H3-LICENSE.txt'} <= names
        assert len([name for name in names if name.endswith('.safetensors')]) == 13
        assert task['requiredAdapter']['path'] == ('ref2va' if workflow == 'ref2va' else 'fl2va') + '/adapter.safetensors'


def test_lookup_returns_independent_data_and_requires_a_workflow():
    first = reviewed_joint_av_artifact_closure(PIKA, 't2va')
    first['files'].clear()
    assert len(reviewed_joint_av_artifact_closure(PIKA, 't2va')['files']) == 35
    with pytest.raises(ValueError, match='exact reviewed'):
        reviewed_joint_av_artifact_closure(PIKA, '')
    assert reviewed_joint_av_artifact_closure('OtherPipeline', 't2va') is None


def test_artifact_descriptor_rejects_changed_identity_partition_paths_digests_and_notices():
    document = json.loads(PATH.read_bytes())
    assert validate_joint_av_artifact_closures(document) == document
    mutations = [
        lambda row: row.__setitem__('revision', 'main'),
        lambda row: row['workflows'][0].__setitem__('transformerPartition', 'transformer_ref'),
        lambda row: row['workflows'][0].__setitem__('selectedByteSize', True),
        lambda row: row['workflows'][0]['files'][0].__setitem__('path', '../outside'),
        lambda row: row['workflows'][0]['files'][0].__setitem__('byteSize', True),
        lambda row: row['workflows'][0]['files'][0].__setitem__('gitBlobId', 'main'),
        lambda row: row['workflows'][0]['requiredAdapter'].__setitem__('sha256', '0' * 64),
        lambda row: row['workflows'][0].__setitem__('files', [file for file in row['workflows'][0]['files'] if file['path'] != 'NOTICE']),
    ]
    for mutate in mutations:
        changed = deepcopy(document)
        mutate(changed['repositories'][1])
        with pytest.raises(ValueError):
            validate_joint_av_artifact_closures(changed)


def test_discovery_uses_no_model_runtime_imports(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'diffusers', 'torch', 'transformers', 'peft'}:
            raise AssertionError('File-closure inspection imported the model stack')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    validate_joint_av_artifact_closures(json.loads(PATH.read_bytes()))
