"""Authorize decode-only recovery for exact native Hunyuan 1.5 workflows."""

from copy import deepcopy
import re
from modiff.hunyuan_video15_artifacts import (
    HUNYUAN_720P_I2V_REPOSITORY,
    HUNYUAN_720P_I2V_REVISION,
    HUNYUAN_720P_T2V_REPOSITORY,
    HUNYUAN_720P_T2V_REVISION,
)
from modiff.material_checkpoints import read_material_checkpoint

_MODULAR = 'modules.ModularDiffusers'
_PIPELINE = 'HunyuanVideo15ModularPipeline'
_DENOISE = 'WorkflowHunyuanVideo15Denoise'
_DECODE = 'WorkflowHunyuanVideo15Decode'
_TEXT = 'WorkflowHunyuanVideo15TextEncode'
_VAE = 'WorkflowHunyuanVideo15VaeEncode'
_IMAGE = 'WorkflowHunyuanVideo15ImageEncode'
_STAGES = {
    _TEXT: 'text_encoder', _VAE: 'vae_encoder', _IMAGE: 'image_encoder', _DENOISE: 'denoise', _DECODE: 'decode',
}
_RECIPES = {
    'text2video': (HUNYUAN_720P_T2V_REPOSITORY, HUNYUAN_720P_T2V_REVISION),
    'image2video': (HUNYUAN_720P_I2V_REPOSITORY, HUNYUAN_720P_I2V_REVISION),
}
_SAFE_HINTS = {'clientRunId', 'runInputHash', 'workflowTabId', 'workflowTitle', 'workflowCanvasEpoch',
               'workflowSnapshot', 'resourceMode'}


class DecodeRecoveryUnavailable(ValueError):
    """No verified decode-only continuation is available for this request."""

    modiff_error_code = 'decode_recovery_unavailable'
    modiff_category = 'runtime'


def _literal(node, key, default=None):
    return node['params'].get(key, {}).get('value', default)


def _links(node):
    return {key: (field['sourceId'], field.get('sourceKey')) for key, field in node['params'].items()
            if field.get('sourceId')}


def _require(condition, message):
    if not condition:
        raise DecodeRecoveryUnavailable(message)


def _review_graph(graph):
    """Identify the only supported closure and reject extra owners, mutations or branches."""
    from modiff.workflow_auto_resource import graph_material

    _require(isinstance(graph, dict) and set(graph) <= {'sid', 'nodes', 'paths', 'runtimeHints'},
             'Decode recovery needs a plain native Hunyuan graph.')
    nodes, paths = graph.get('nodes'), graph.get('paths')
    hints = graph.get('runtimeHints') or {}
    _require(isinstance(nodes, dict) and len(nodes) in {5, 8} and isinstance(paths, list) and len(paths) == 1
             and isinstance(hints, dict) and set(hints) <= _SAFE_HINTS and hints.get('resourceMode') != 'auto',
             'Decode recovery supports one native Hunyuan Expert workflow at a time.')
    graph_material(graph)
    _require(len(paths[0]) == len(nodes), 'Decode recovery cannot resume repeated nodes.')

    ids = {}
    for node_id in paths[0]:
        node = nodes[node_id]
        identity = (node['module'], node['action'])
        if identity == ('modules.Video', 'ExportAsset'):
            role = 'output'
        elif identity == ('modules.Video', 'Export'):
            raise DecodeRecoveryUnavailable('Decode recovery needs the app-managed retained video output.')
        elif identity == ('modules.Image', 'Load'):
            role = 'image'
        elif identity == (_MODULAR, 'ModelsLoader'):
            role = 'loader'
        elif identity[0] == _MODULAR and identity[1] in _STAGES:
            role = identity[1]
        else:
            raise DecodeRecoveryUnavailable('Decode recovery does not execute unreviewed or custom nodes.')
        _require(role not in ids, 'Decode recovery cannot resume a graph with more than one owner or output per stage.')
        ids[role] = node_id

    workflow = 'text2video' if len(nodes) == 5 else 'image2video'
    chain = [_TEXT, _DENOISE, _DECODE] if workflow == 'text2video' else [_TEXT, _VAE, _IMAGE, _DENOISE, _DECODE]
    expected = {'loader', *chain, 'output'}
    if workflow == 'image2video':
        expected.add('image')
    _require(set(ids) == expected and paths[0][-2:] == [ids[_DECODE], ids['output']],
             'Decode recovery could not establish the reviewed Hunyuan state chain.')

    repo, revision = _RECIPES[workflow]
    loader_id = ids['loader']
    loader = nodes[loader_id]
    _require(not _links(loader) and _literal(loader, 'model_type') == _PIPELINE
             and _literal(loader, 'workflow_id') == workflow
             and _literal(loader, 'repo_id') == {'source': 'hub', 'value': repo}
             and _literal(loader, 'revision') == revision and not _literal(loader, 'trust_remote_code')
             and not _literal(loader, 'lora_list') and not _literal(loader, 'quant_config'),
             'Decode recovery requires the original reviewed Hub model and loader settings.')
    dtype = _literal(loader, 'dtype')
    _require(dtype in {'float16', 'bfloat16', 'float32'}, 'Decode recovery requires an explicit model dtype.')

    previous = None
    for role in chain:
        node = nodes[ids[role]]
        dependencies = {'pipeline_components': (loader_id, 'pipeline_components')}
        if previous:
            dependencies['state_in'] = (ids[previous], 'state_out')
        if role == _VAE:
            dependencies['image'] = (ids['image'], 'image')
        _require(_links(node) == dependencies and _literal(node, 'pipeline_class') == _PIPELINE
                 and _literal(node, 'workflow_id') == workflow and _literal(node, 'block_path') == _STAGES[role],
                 'Decode recovery requires the exact original loader-to-stage bindings.')
        previous = role
    if 'image' in ids:
        _require(not _links(nodes[ids['image']]), 'Decode recovery only supports the recorded source image closure.')
    _require(_links(nodes[ids['output']]) == {'video': (ids[_DECODE], 'video')},
             'Decode recovery cannot restore a video connected to other output inputs.')
    return workflow, ids, dtype


def _geometry(record, graph, workflow, ids):
    tensor = record['tensor']
    shape = tensor['shape']
    _require(isinstance(shape, list) and len(shape) == 5 and all(type(k) is int and k > 0 for k in shape)
             and shape[:2] == [1, 32],
             'The retained denoise tensor does not have the reviewed batch and latent channels.')
    # The reviewed full 720p checkpoint uses a 4× temporal and 16× spatial VAE grid.
    frames, height, width = (shape[2] - 1) * 4 + 1, shape[3] * 16, shape[4] * 16
    nodes = graph['nodes']
    _require(5 <= frames <= 481 and 256 <= height <= 2048 and 256 <= width <= 2048
             and int(_literal(nodes[ids[_DENOISE]], 'num_frames', 121)) == frames,
             'The retained tensor does not match the original frame count and geometry.')
    geometry_owner = nodes[ids[_DENOISE] if workflow == 'text2video' else ids[_VAE]]
    requested = [_literal(geometry_owner, 'width'), _literal(geometry_owner, 'height')]
    if workflow == 'text2video':
        requested = [value if value is not None else default for value, default in zip(requested, [848, 480])]
    _require(requested == [None, None] or None not in requested and list(map(int, requested)) == [width, height],
             'The retained tensor does not match the original requested dimensions.')
    return {'frames': frames, 'height': height, 'width': width}


def _image_identity(server, graph, image_id):
    from modiff.builtin_media_evidence import file_identity

    value = _literal(graph['nodes'][image_id], 'file')
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    _require(isinstance(value, str) and not value.startswith(('http://', 'https://')),
             'Decode recovery needs the original managed local source image.')
    path = server._resolve_managed_path_identifier(value)
    _require(path is not None and path.is_file(), 'The original source image is no longer available.')
    return file_identity(path)


def describe_completed_sampling(server, task, record, state):
    """Record the exact accepted graph, not a supplied workflow snapshot or advisory form."""
    graph = getattr(server, 'task_graphs', {}).get(task['task_id'])
    try:
        from modiff.workflow_auto_resource import graph_material

        workflow, ids, dtype = _review_graph(graph)
        if (ids[_DENOISE] != record['sourceNodeId'] or record['workflowId'] != workflow
                or (record['repository'], record['revision']) != _RECIPES[workflow]
                or record['tensor']['dtype'] != 'torch.' + dtype
                or not record['runInputHash'] or not record['runtimeFingerprint'] or not record['backendSource']
                or record['runInputHash'] != (graph.get('runtimeHints') or {}).get('runInputHash')
                or task.get('attempt_index', 0) != 0
                or graph_material(task['args'][0]) != graph_material(graph)):
            return None
        geometry = _geometry(record, graph, workflow, ids)
        for field, key in (('width', 'width'), ('height', 'height'), ('num_frames', 'frames')):
            actual = state.get(field)
            if actual is not None and type(actual) is int and actual != geometry[key]:
                return None
        recovery = {'graph': deepcopy(graph), 'decodeNodeId': ids[_DECODE], 'geometry': geometry}
        if 'image' in ids:
            from modules.Image.main import Load

            source_node = server.node_cache.get(ids['image'])
            source_hash = source_node.output.get('source_hash') if type(source_node) is Load else None
            if not isinstance(source_hash, str) or not re.fullmatch(r'[a-f0-9]{64}', source_hash):
                return None
            image = _image_identity(server, graph, ids['image'])
            if image['sha256'] != source_hash:
                return None
            recovery['sourceImage'] = image
        return recovery
    except DecodeRecoveryUnavailable as exc:
        task['_decode_recovery_blocker'] = str(exc)
        return None
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        return None


def is_failed_native_decode(task):
    return task.get('status') == 'failed' and task.get('node_name') == f'{_MODULAR}.{_DECODE}'


def recordable_decode_failure(task, entry):
    anchor = task.get('_material_checkpoint')
    if (isinstance(anchor, dict) and is_failed_native_decode(entry)
            and entry.get('node') == anchor.get('decodeNodeId')
            and entry.get('attempt_index') == anchor.get('attemptIndex')):
        return anchor
    return None


def _read_retry(server, task, *, load_tensor=False, verify_tensor=False):
    anchor = task.get('decodeRecoveryCheckpoint') if isinstance(task, dict) else None
    blocker = task.get('decodeRecoveryBlocker') if isinstance(task, dict) else None
    if isinstance(blocker, str) and blocker:
        raise DecodeRecoveryUnavailable(blocker)
    _require(task and is_failed_native_decode(task) and isinstance(anchor, dict),
             'No verified completed sampling is available for this failed decode.')
    _require(not any(item.get('status') == 'completed' and item.get('recoveredFromTaskId') == task['task_id']
                     for item in getattr(server, 'recent_tasks', ())),
             'A retry already produced a video for this run.')
    try:
        result = read_material_checkpoint(
            server.data_dir, task_id=task['task_id'], source_node_id=anchor['sourceNodeId'],
            receipt_sha256=anchor['receiptSHA256'], load_tensor=load_tensor, verify_tensor=verify_tensor,
        )
        record, tensor = result if load_tensor else (result, None)
        recovery = record['recovery']
        graph = recovery['graph']
        workflow, ids, dtype = _review_graph(graph)
        _require(record['backendSource'] == server.backend_source_identity.get('fingerprint'),
                 'The backend source changed since sampling. Run the workflow again.')
        _require(record['runtimeFingerprint'] == task.get('runtimeFingerprint')
                 and record['runInputHash'] == task.get('run_input_hash')
                 and record['runInputHash'] == (graph.get('runtimeHints') or {}).get('runInputHash')
                 and record['workflowId'] == workflow and record['tensor']['dtype'] == 'torch.' + dtype
                 and (record['repository'], record['revision']) == _RECIPES[workflow]
                 and anchor['decodeNodeId'] == recovery['decodeNodeId'] == ids[_DECODE] == task['node']
                 and record['sourceNodeId'] == ids[_DENOISE] and record['completedStage'] == 'denoise'
                 and anchor['attemptIndex'] == task.get('attempt_index')
                 and recovery['geometry'] == _geometry(record, graph, workflow, ids),
                 'The checkpoint no longer matches the exact failed decode and retained inputs.')
        tiling = _literal(graph['nodes'][ids[_DECODE]], 'vae_tiling')
        _require(tiling is None or type(tiling) is bool, 'The original VAE tiling setting is invalid.')
        if 'image' in ids:
            _require(recovery.get('sourceImage') == _image_identity(server, graph, ids['image']),
                     'The original source image changed since sampling. Run the workflow again.')
        return record, tensor, ids, tiling
    except DecodeRecoveryUnavailable:
        raise
    except (OSError, KeyError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise DecodeRecoveryUnavailable('The retained sampling files are missing, expired or invalid.') from exc


def decode_recovery_status(server, task):
    if not is_failed_native_decode(task):
        return None
    status = {'kind': 'decode', 'status': 'unavailable', 'vaeTiling': None, 'canEnableVaeTiling': False}
    try:
        record, _, _, tiling = _read_retry(server, task, verify_tensor=True)
        runtime = getattr(server, '_last_runtime_fingerprint', None)
        _require(isinstance(runtime, dict) and runtime.get('fingerprint') == record['runtimeFingerprint'],
                 'The execution runtime changed since sampling. Run the workflow again.')
    except DecodeRecoveryUnavailable as exc:
        return {**status, 'reason': str(exc)}
    return {**status, 'status': 'ready', 'vaeTiling': tiling,
            'canEnableVaeTiling': task.get('oom') is True and tiling is not True}


def queueable_retry(server, task, sid, enable_tiling):
    record, _, _, tiling = _read_retry(server, task, verify_tensor=True)
    _require(not enable_tiling or (task.get('oom') is True and tiling is not True),
             'Explicit tiled retry is available only after an untiled decode ran out of memory.')
    runtime = server._runtime_fingerprint_for_control_request()
    _require(runtime.get('fingerprint') == record['runtimeFingerprint'],
             'The execution runtime changed since sampling. Run the workflow again.')
    graph = deepcopy(record['recovery']['graph'])
    graph['sid'] = sid
    # Keep the generating run identity separate from the replacement queue task.
    graph['runtimeHints'].pop('clientRunId', None)
    return graph


def prepare_retry(server, graph, request, fingerprint):
    """Revalidate all bindings before the queue may dispatch its first model."""
    source_id = request['sourceTaskId']
    task = next((item for item in server.recent_tasks if item.get('task_id') == source_id), None)
    record, tensor, ids, original_tiling = _read_retry(server, task, load_tensor=True)
    from modiff.workflow_auto_resource import graph_material
    _require(graph_material(graph) == graph_material(record['recovery']['graph'])
             and fingerprint == record['runtimeFingerprint'],
             'The queued graph or runtime no longer matches the retained sampling.')
    enable_tiling = request['enableVaeTiling']
    _require(not enable_tiling or (task.get('oom') is True and original_tiling is not True),
             'The original run does not authorize the requested tiled retry.')
    return {
        'record': record, 'tensor': tensor, 'loaderId': ids['loader'], 'decodeId': ids[_DECODE],
        'path': [ids['loader'], ids[_DECODE], ids['output']],
        'tiling': True if enable_tiling else original_tiling, 'originalTiling': original_tiling,
    }


def restored_decoder_node(server, node, plan):
    """Bind material to a fresh, normal loader token for one built-in decoder dispatch."""
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from modules.ModularDiffusers.loaders import ModelsLoader
    from modules.ModularDiffusers.route_state import require_component_binding
    from modules.ModularDiffusers.workflow_blocks import _issue_workflow_state

    loader = server.node_cache[plan['loaderId']]
    _require(type(loader) is ModelsLoader, 'The retry did not load the reviewed built-in model owner.')
    token = require_component_binding(loader.output.get('pipeline_components'), label='decode retry model',
                                     expected_model_type=_PIPELINE, expected_role='pipeline_components')
    record = plan['record']
    _require((token._repo_source, token._repo_id, token._revision) ==
             ('hub', record['repository'], record['revision']),
             'The retry loader no longer belongs to the exact captured checkpoint.')
    latents = plan['tensor'].to(device=record['tensor']['sourceDevice'])
    state = _issue_workflow_state(token=token, pipeline_class=_PIPELINE, workflow_id=record['workflowId'],
                                  completed_stage='denoise', state=PipelineState(values={'latents': latents}))
    transient = deepcopy(node)
    transient['params']['state_in'] = {'value': state}
    transient['params']['vae_tiling'] = {'value': plan['tiling']}
    return transient
