"""Finite immutable file closures; discovery grants no download or execution."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path, PurePosixPath
import re


PATH = Path(__file__).resolve().parents[1] / 'data/joint-av-artifact-closures.v1.json'
_IDENTITIES = {
    'MiniMaxH3ModularPipeline': ('MiniMaxAI/MiniMax-H3', '42ed227ee7df40d41602854ae760620d6eb651fe'),
    'PikaH3FlexModularPipeline': ('Pika-Labs/Pika-H3-Flex', '3ff166447d5edb1406107ab8370c42000ce173e4'),
}
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_GIT = re.compile(r'[0-9a-f]{40}\Z')


def validate_joint_av_artifact_closures(document):
    if (type(document) is not dict or set(document) != {'schemaVersion', 'kind', 'scope', 'repositories'}
            or type(document['schemaVersion']) is not int or document['schemaVersion'] != 1
            or document['kind'] != 'reviewed-joint-av-artifact-closures'
            or type(document['scope']) is not str or len(document['scope']) > 512
            or type(document['repositories']) is not list or len(document['repositories']) != 2):
        raise ValueError('Malformed reviewed joint AV artifact closures.')
    rows = {}
    for row in document['repositories']:
        if (type(row) is not dict or set(row) != {'pipelineClass', 'repository', 'revision', 'license', 'workflows'}
                or row['pipelineClass'] not in _IDENTITIES or row['pipelineClass'] in rows
                or (row['repository'], row['revision']) != _IDENTITIES[row['pipelineClass']]
                or row['license'] != {'path': 'LICENSE', 'sha256':
                    '59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44'}
                or type(row['workflows']) is not list or len(row['workflows']) != 3):
            raise ValueError('Unreviewed joint AV model identity or license descriptor.')
        workflows = {}
        for task in row['workflows']:
            if (type(task) is not dict or set(task) != {'workflowId', 'transformerPartition', 'requiredAdapter', 'files', 'selectedByteSize'}
                    or task['workflowId'] not in {'t2va', 'fl2va', 'ref2va'} or task['workflowId'] in workflows
                    or type(task['files']) is not list or not 1 <= len(task['files']) <= 128):
                raise ValueError('Malformed joint AV workflow file selection.')
            partition = 'transformer_ref' if task['workflowId'] == 'ref2va' else 'transformer'
            if task['transformerPartition'] != partition:
                raise ValueError('Joint AV artifact partition does not match its selected task.')
            names, byte_size = set(), 0
            for file in task['files']:
                if (type(file) is not dict or set(file) != {'path', 'byteSize', 'sha256', 'gitBlobId'}
                        or type(file['path']) is not str or not 1 <= len(file['path']) <= 256):
                    raise ValueError('Malformed joint AV file descriptor.')
                name = file['path']
                pure = PurePosixPath(name)
                if (pure.is_absolute() or '\\' in name or ':' in name or pure.as_posix() != name
                        or any(part in {'.', '..'} for part in pure.parts) or name in names
                        or type(file['byteSize']) is not int or not 0 < file['byteSize'] <= 100 * 1024**3
                        or type(file['gitBlobId']) is not str or _GIT.fullmatch(file['gitBlobId']) is None
                        or file['sha256'] is not None and (type(file['sha256']) is not str or _SHA.fullmatch(file['sha256']) is None)
                        or name.endswith('.safetensors') and file['sha256'] is None):
                    raise ValueError('Unsafe joint AV file path, size or immutable digest.')
                if name.startswith(('transformer/' if partition == 'transformer_ref' else 'transformer_ref/', 'FL2VA/', 'Ref2VA/')):
                    raise ValueError('The joint AV selection contains another transformer partition.')
                if (name not in {'LICENSE', 'NOTICE'}
                        and not name.endswith(('.json', '.txt', '.model', '.jinja', '.tiktoken', '.safetensors'))):
                    raise ValueError('The joint AV closure includes an unreviewed file kind.')
                names.add(name)
                byte_size += file['byteSize']
            if (not {'LICENSE', 'modular_model_index.json'} <= names
                    or type(task['selectedByteSize']) is not int or task['selectedByteSize'] != byte_size):
                raise ValueError('The joint AV file closure is incomplete or has an invalid byte total.')
            from modiff.joint_av_model_adapters import required_joint_av_adapter
            adapter = required_joint_av_adapter(row['pipelineClass'], row['repository'], row['revision'], task['workflowId'])
            if adapter is None:
                if task['requiredAdapter'] is not None:
                    raise ValueError('An unadapted joint AV model cannot declare a required adapter.')
            else:
                expected = {'repository': row['repository'], 'revision': row['revision'],
                            'path': adapter.weight_name, 'byteSize': adapter.byte_size, 'sha256': adapter.sha256}
                if (task['requiredAdapter'] != expected or adapter.weight_name not in names
                        or not {'NOTICE', 'licenses/MiniMax-H3-LICENSE.txt'} <= names):
                    raise ValueError('The required Pika adapter or derivative notices are incomplete.')
                file = next(file for file in task['files'] if file['path'] == adapter.weight_name)
                if file['sha256'] != adapter.sha256 or file['byteSize'] != adapter.byte_size:
                    raise ValueError('Required Pika adapter bytes differ from the reviewed model recipe.')
            adapter_files = {name for name in names if name.startswith(('fl2va/', 'ref2va/'))}
            if adapter_files != ({adapter.weight_name} if adapter is not None else set()):
                raise ValueError('The joint AV closure includes an adapter for another task.')
            workflows[task['workflowId']] = task
        rows[row['pipelineClass']] = row
    return document


@lru_cache(maxsize=1)
def _closures():
    with PATH.open('rb') as stream:
        raw = stream.read(512 * 1024 + 1)
    if len(raw) > 512 * 1024:
        raise ValueError('Joint AV artifact descriptor exceeds its finite bound.')
    return validate_joint_av_artifact_closures(json.loads(raw))


def reviewed_joint_av_artifact_closure(pipeline_class, workflow_id):
    if pipeline_class not in _IDENTITIES:
        return None
    row = next(row for row in _closures()['repositories'] if row['pipelineClass'] == pipeline_class)
    task = next((task for task in row['workflows'] if task['workflowId'] == workflow_id), None)
    if task is None:
        raise ValueError('Select an exact reviewed joint AV workflow before its model files.')
    return deepcopy({'pipelineClass': pipeline_class, 'repository': row['repository'], 'revision': row['revision'], **task})


def reviewed_joint_av_artifact_selections(pipeline_class):
    """Publish ordinary app inventories without merging task partitions."""
    if pipeline_class not in _IDENTITIES:
        return []
    from modiff.supplemental_modular_contracts import PIKA_WORKFLOW_TASKS
    labels = {'t2va': 'Text to video with audio', 'fl2va': 'First/last frame to video with audio',
              'ref2va': 'Reference to video with audio'}
    result = []
    for workflow, mode in PIKA_WORKFLOW_TASKS.items():
        task = reviewed_joint_av_artifact_closure(pipeline_class, workflow)
        result.append({'modes': [mode], 'repo': task['repository'], 'revision': task['revision'],
                       'downloadFiles': [file['path'] for file in task['files']], 'label': labels[workflow]})
    return result


def reviewed_joint_av_download_selection(repository, revision, files):
    """Validate explicit app acquisition through the existing Hub downloader.

    Text and keyframe routes share the same exact bytes. A matching complete
    selection admits those files, not an inferred execution task or runtime.
    """
    pipeline = next((name for name, (repo, _pin) in _IDENTITIES.items() if repo == repository), None)
    if pipeline is None:
        return None
    selections = reviewed_joint_av_artifact_selections(pipeline)
    if revision not in (None, _IDENTITIES[pipeline][1]):
        raise ValueError('This model requires its exact reviewed immutable revision.')
    for selection in selections:
        if sorted(files) == sorted(selection['downloadFiles']):
            return deepcopy(selection)
    raise ValueError('Choose an exact reviewed workflow file selection; whole-repository, partial, and mixed-task downloads are not admitted.')
