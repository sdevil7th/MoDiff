"""One bounded Hunyuan decode retry through the ordinary queue and native VAE."""

import asyncio
from collections import Counter
from copy import deepcopy
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from modiff import material_checkpoints as storage
from modiff.hunyuan_video15_artifacts import (
    HUNYUAN_720P_I2V_REPOSITORY, HUNYUAN_720P_I2V_REVISION,
    HUNYUAN_720P_T2V_REPOSITORY, HUNYUAN_720P_T2V_REVISION,
)
from modiff.material_decode_recovery import DecodeRecoveryUnavailable, decode_recovery_status, prepare_retry, queueable_retry
from modules.ModularDiffusers.route_state import bind_loader_outputs, issue_pipeline_instance_token, require_component_binding
from modules.ModularDiffusers.workflow_blocks import WorkflowHunyuanVideo15Denoise, _issue_workflow_state

PIPELINE = 'HunyuanVideo15ModularPipeline'
MODULE = 'modules.ModularDiffusers'
RUNTIME = 'sha256:' + '1' * 64
SOURCE = 'sha256:' + '2' * 64


def _wire(node, field):
    return {'sourceId': node, 'sourceKey': field}


def _node(action, *, module=MODULE, wires=None, **values):
    return {'module': module, 'action': action,
            'params': {**{key: {'value': value} for key, value in values.items()}, **(wires or {})}}


def _graph(tiling=None):
    loader = _wire('loader', 'pipeline_components')
    common = {'pipeline_class': PIPELINE, 'workflow_id': 'text2video'}
    nodes = {
        'loader': _node('ModelsLoader', model_type=PIPELINE, workflow_id='text2video', dtype='float32',
                        device='cpu', offload_mode='none', trust_remote_code=False,
                        repo_id={'source': 'hub', 'value': HUNYUAN_720P_T2V_REPOSITORY},
                        revision=HUNYUAN_720P_T2V_REVISION),
        'text': _node('WorkflowHunyuanVideo15TextEncode', wires={'pipeline_components': loader},
                      **common, block_path='text_encoder', prompt='An exact prompt'),
        'sampler': _node('WorkflowHunyuanVideo15Denoise', wires={'pipeline_components': loader,
                         'state_in': _wire('text', 'state_out')}, **common, block_path='denoise',
                         width=256, height=256, num_frames=5, num_inference_steps=1, seed=7),
        'decode': _node('WorkflowHunyuanVideo15Decode', wires={'pipeline_components': loader,
                        'state_in': _wire('sampler', 'state_out')}, **common, block_path='decode',
                        vae_tiling=tiling),
        'output': _node('ExportAsset', module='modules.Video', wires={'video': _wire('decode', 'video')}),
    }
    nodes['output']['params']['preview'] = {'display': 'ui_video', 'sourceKey': 'file'}
    return {'sid': 'original', 'nodes': nodes, 'paths': [list(nodes)],
            'runtimeHints': {'runInputHash': 'run_exact', 'clientRunId': 'original-client-run'}}


def _token(repo=HUNYUAN_720P_T2V_REPOSITORY, revision=HUNYUAN_720P_T2V_REVISION):
    return issue_pipeline_instance_token(model_type=PIPELINE, repo_id=repo, repo_source='hub', revision=revision)


def _capture(root, graph=None, source=None):
    graph = _graph() if graph is None else graph
    loader = graph['nodes']['loader']['params']
    task = {'task_id': 'original', 'runtimeFingerprint': RUNTIME, 'args': [graph],
            'runtimeHints': graph['runtimeHints'], 'attempt_index': 0}
    node = WorkflowHunyuanVideo15Denoise('sampler')
    node._has_changed = True
    latent = torch.linspace(-1, 1, 32 * 2 * 16 * 16).reshape(1, 32, 2, 16, 16)
    node.output = {'state_out': _issue_workflow_state(
        token=_token(loader['repo_id']['value']['value'], loader['revision']['value']), pipeline_class=PIPELINE,
        workflow_id=loader['workflow_id']['value'], completed_stage='denoise',
        state={'latents': latent})}
    server = SimpleNamespace(data_dir=root, task_graphs={'original': deepcopy(graph)}, current_task=task,
                             node_cache={'sampler': node}, backend_source_identity={'fingerprint': SOURCE},
                             _last_runtime_fingerprint={'fingerprint': RUNTIME}, recent_tasks=[])
    if source is not None:
        from modiff.path_identifiers import resolve_managed_path_identifier

        server.node_cache['source'] = source
        server._resolve_managed_path_identifier = lambda value: resolve_managed_path_identifier(
            value, work_root=root, data_root=root)
    record = storage.capture_completed_material(server, 'sampler')
    return server, record, latent


def _failed_entry(server):
    return {'task_id': 'original', 'status': 'failed', 'node': 'decode',
            'node_name': f'{MODULE}.WorkflowHunyuanVideo15Decode', 'attempt_index': 0,
            'decodeRecoveryCheckpoint': server.current_task['_material_checkpoint'],
            'runtimeFingerprint': RUNTIME, 'run_input_hash': 'run_exact', 'oom': True}


def test_material_storage_enforces_pair_count_bytes_and_expiration(tmp_path, monkeypatch):
    root = tmp_path / storage.CHECKPOINT_SUBDIRECTORY
    root.mkdir(parents=True)
    now = time.time()

    def put(key, age=0):
        for suffix in ('.json', '.safetensors'):
            path = root / (key * 64 + suffix)
            path.write_bytes(b'{}' if suffix == '.json' else b'0' * 20)
            os.utime(path, (now - age, now - age))

    put('a', storage.RETENTION_SECONDS + 10)
    put('b', 30)
    put('c', 20)
    put('d', 10)
    (root / ('e' * 64 + '.safetensors')).write_bytes(b'orphan')
    (root / '.checkpoint-abandoned.tmp').write_bytes(b'partial transaction')
    (root / 'operator-file.txt').write_bytes(b'untouched')
    monkeypatch.setattr(storage, 'MAX_STORED_BYTES', 43)
    monkeypatch.setattr(storage, 'MAX_STORED_PAIRS', 2)
    storage.prune_material_checkpoints(tmp_path)
    assert sorted(p.name for p in root.iterdir()) == [
        'd' * 64 + '.json', 'd' * 64 + '.safetensors', 'operator-file.txt']


@pytest.mark.parametrize('case', ['tensor', 'missing', 'receipt', 'different-input', 'different-runtime',
                                  'different-backend', 'different-task', 'expired', 'wrong-recipe'])
def test_retry_rejects_stale_corrupt_or_foreign_material_before_loading(tmp_path, case):
    graph = _graph()
    server, record, _ = _capture(tmp_path)
    failed = _failed_entry(server)
    server.recent_tasks = [failed]
    root = tmp_path / storage.CHECKPOINT_SUBDIRECTORY
    tensor_path = root / record['tensorFile']
    metadata_path = next(root.glob('*.json'))
    fingerprint = RUNTIME
    source_task = 'original'
    if case == 'tensor':
        body = tensor_path.read_bytes()
        tensor_path.write_bytes(body[:-1] + bytes([body[-1] ^ 1]))
        assert decode_recovery_status(server, failed)['status'] == 'unavailable'
        with pytest.raises(DecodeRecoveryUnavailable, match='missing, expired or invalid'):
            queueable_retry(server, failed, 'sid', False)
    elif case == 'missing':
        tensor_path.unlink()
    elif case == 'receipt':
        record['recovery']['graph']['nodes']['sampler']['params']['seed']['value'] = 8
        metadata_path.write_text(json.dumps(record))
    elif case == 'different-input':
        failed['run_input_hash'] = 'run_different'
    elif case == 'different-runtime':
        fingerprint = 'sha256:' + '3' * 64
    elif case == 'different-backend':
        server.backend_source_identity['fingerprint'] = 'sha256:' + '4' * 64
    elif case == 'different-task':
        failed['task_id'] = source_task = 'another-task'
    elif case == 'expired':
        old_time = time.time() - storage.RETENTION_SECONDS - 1
        os.utime(metadata_path, (old_time, old_time))
    elif case == 'wrong-recipe':
        graph['nodes']['sampler']['params']['seed']['value'] = 8
    with pytest.raises(DecodeRecoveryUnavailable):
        prepare_retry(server, graph, {'sourceTaskId': source_task, 'enableVaeTiling': True}, fingerprint)


@pytest.mark.parametrize('case', ['custom', 'explicit-export', 'different-owner', 'foreign-repo', 'auto', 'loop',
                                  'wrong-geometry'])
def test_unsupported_capture_is_archival_only(tmp_path, case):
    graph = _graph()
    if case == 'custom':
        graph['nodes']['output']['module'] = 'modules.ThirdParty'
    elif case == 'explicit-export':
        graph['nodes']['output']['action'] = 'Export'
    elif case == 'different-owner':
        graph['nodes']['decode']['params']['pipeline_components']['sourceId'] = 'text'
    elif case == 'foreign-repo':
        graph['nodes']['loader']['params']['repo_id']['value']['value'] = 'other/repo'
    elif case == 'auto':
        graph['runtimeHints']['resourceMode'] = 'auto'
    elif case == 'loop':
        graph['loops'] = []
    elif case == 'wrong-geometry':
        graph['nodes']['sampler']['params']['width']['value'] = 272
    server, record, _ = _capture(tmp_path, graph)
    assert '_material_checkpoint' not in server.current_task
    assert record['schemaVersion'] == 1 and record['restorationAuthorized'] is False
    if case == 'explicit-export':
        assert 'app-managed retained video output' in server.current_task['_decode_recovery_blocker']


def test_failed_decoder_reports_inherited_tiling_and_verified_retry_policy(tmp_path):
    app, _, _ = _capture(tmp_path)
    failed = _failed_entry(app)
    assert decode_recovery_status(app, failed) == {
        'kind': 'decode', 'status': 'ready', 'vaeTiling': None, 'canEnableVaeTiling': True,
    }
    failed['oom'] = False
    assert decode_recovery_status(app, failed)['canEnableVaeTiling'] is False
    app.recent_tasks = [failed]
    with pytest.raises(DecodeRecoveryUnavailable, match='tiled retry'):
        prepare_retry(app, _graph(), {'sourceTaskId': 'original', 'enableVaeTiling': True}, RUNTIME)


def test_image_to_video_retry_revalidates_the_native_source_image_bytes(tmp_path):
    from PIL import Image
    from modules.Image.main import Load

    file = tmp_path / 'original.png'
    Image.new('RGB', (256, 256), 'red').save(file)
    source = Load('source')
    source.output = source.execute(file=[str(file)])
    graph = _graph()
    nodes = graph['nodes']
    common = {'pipeline_class': PIPELINE, 'workflow_id': 'image2video'}
    for stage in ('loader', 'text', 'sampler', 'decode'):
        nodes[stage]['params']['workflow_id']['value'] = 'image2video'
    nodes['loader']['params']['repo_id']['value']['value'] = HUNYUAN_720P_I2V_REPOSITORY
    nodes['loader']['params']['revision']['value'] = HUNYUAN_720P_I2V_REVISION
    nodes['source'] = _node('Load', module='modules.Image', file=[str(file)])
    loader = _wire('loader', 'pipeline_components')
    nodes['reference'] = _node('WorkflowHunyuanVideo15VaeEncode', **common, block_path='vae_encoder',
                               wires={'pipeline_components': loader, 'state_in': _wire('text', 'state_out'),
                                      'image': _wire('source', 'image')})
    nodes['embed'] = _node('WorkflowHunyuanVideo15ImageEncode', **common, block_path='image_encoder',
                           wires={'pipeline_components': loader, 'state_in': _wire('reference', 'state_out')})
    nodes['sampler']['params']['state_in'] = _wire('embed', 'state_out')
    graph['paths'] = [['loader', 'text', 'source', 'reference', 'embed', 'sampler', 'decode', 'output']]
    app, record, _ = _capture(tmp_path, graph, source)
    failed = _failed_entry(app)
    assert record['recovery']['sourceImage']['sha256'] == source.output['source_hash']
    assert decode_recovery_status(app, failed)['status'] == 'ready'
    Image.new('RGB', (256, 256), 'blue').save(file)
    status = decode_recovery_status(app, failed)
    assert status['status'] == 'unavailable' and 'source image changed' in status['reason']


def test_material_store_is_unreachable_through_public_file_resolver(tmp_path):
    from modiff.path_identifiers import resolve_runtime_input_path
    from modiff.server import WebServer

    app = object.__new__(WebServer)
    app.work_dir = app.data_dir = str(tmp_path)
    private = tmp_path / storage.CHECKPOINT_SUBDIRECTORY
    private.mkdir(parents=True)
    (private / 'test.json').write_text('{}')
    assert app._resolve_managed_path_identifier('@data/runtime/material-checkpoints/test.json') is None
    assert app._resolve_managed_path_identifier(private / 'test.json') is None
    assert app._resolve_managed_path_identifier('@data/videos/test.mp4') == tmp_path / 'videos/test.mp4'
    for value in [private / 'test.json', '@data/runtime/material-checkpoints/test.json']:
        with pytest.raises(ValueError, match='private MoDiff run state'):
            resolve_runtime_input_path(value, work_root=tmp_path, data_root=tmp_path)

    async def browse():
        query = {'basepath': 'data', 'path': '@data/runtime/material-checkpoints'}
        assert (await app.listdir(SimpleNamespace(query=query))).status == 403
        query['path'] = '@data/runtime'
        response = await app.listdir(SimpleNamespace(query=query))
        assert response.status == 200 and json.loads(response.text)['files'] == []

    asyncio.run(browse())


def test_retry_http_rejects_corrupt_busy_missing_and_malformed_requests(tmp_path):
    from modiff.server import WebServer

    fixture, record, _ = _capture(tmp_path)
    server = object.__new__(WebServer)
    server.__dict__.update(vars(fixture))
    server.recent_tasks = [_failed_entry(fixture)]
    server.queued_tasks = {}
    server._decode_recovery_submissions = set()
    tensor_path = tmp_path / storage.CHECKPOINT_SUBDIRECTORY / record['tensorFile']
    tensor = tensor_path.read_bytes()
    tensor_path.write_bytes(tensor[:-1] + bytes([tensor[-1] ^ 1]))

    async def post(source='original', body=b'{"sid":"retry"}'):
        request = SimpleNamespace(match_info={'task_id': source}, read=lambda: asyncio.sleep(0, result=body))
        return await server.retry_decode(request)

    async def scenario():
        assert (await post(body=b'{"sid":"retry","nodes":{}}')).status == 400
        assert (await post(source='missing')).status == 404
        response = await post()
        assert response.status == 409
        assert json.loads(response.text)['error_code'] == 'decode_recovery_unavailable'
        server.queued_tasks['active'] = {'decode_recovery': {'sourceTaskId': 'original'}}
        response = await post()
        assert response.status == 409 and 'already queued or running' in response.text

    asyncio.run(scenario())


def _run_native_vae_worker(tmp_path, monkeypatch, source_id=None):
    """Run the real decoder and exporter. Model loading and sampling use tiny fixtures."""
    from PIL import Image
    from diffusers import AutoencoderKLHunyuanVideo15, ComponentsManager
    import modules
    from modiff.NodeBase import memory_manager
    from modiff.config import CONFIG
    from modiff.server import WebServer
    from modules.ModularDiffusers import loaders, workflow_blocks as blocks
    from modules.Video.main import ExportAsset
    from utils import huggingface

    for field in ('work_dir', 'data', 'images', 'videos', 'audio', 'models', 'upscalers', 'temp'):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    monkeypatch.setitem(CONFIG.paths, 'temp', str(tmp_path / 'data' / 'temp'))
    monkeypatch.setattr(memory_manager, 'cache', {})
    monkeypatch.setattr(memory_manager, '_active_ids', Counter())
    supervisor_state = tmp_path / 'queue.json'
    monkeypatch.setenv('MODIFF_SUPERVISOR_QUEUE_STATE', str(supervisor_state))
    manager = ComponentsManager()
    monkeypatch.setattr(blocks, 'components', manager)
    monkeypatch.setattr(huggingface, '_hf_cache_locations', lambda: [('unit cache', str(tmp_path))])
    index = Path(__file__).with_name('fixtures') / 'hunyuanvideo15_720p_t2v_index.v1.json'
    snapshot = tmp_path / ('models--' + HUNYUAN_720P_T2V_REPOSITORY.replace('/', '--'))
    snapshot = snapshot / 'snapshots' / HUNYUAN_720P_T2V_REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / 'model_index.json').write_bytes(index.read_bytes())

    def cached_index(repository, revision, filename):
        assert (repository, revision) == (HUNYUAN_720P_T2V_REPOSITORY, HUNYUAN_720P_T2V_REVISION)
        if filename != 'model_index.json':
            raise FileNotFoundError(filename)
        return snapshot

    monkeypatch.setattr(loaders, 'exact_cached_snapshot_path', cached_index)
    vae = AutoencoderKLHunyuanVideo15(block_out_channels=(32,) * 5, layers_per_block=1).eval()
    real_decode = vae.decode
    calls = []

    def decode(latents, **kwargs):
        calls.append(('vae', vae.use_tiling, tuple(latents.shape)))
        if source_id is None:
            raise torch.OutOfMemoryError('Injected allocation fault in the native decoder')
        return real_decode(latents, **kwargs)

    monkeypatch.setattr(vae, 'decode', decode)
    model_id = manager.add('vae', vae, collection='unit-hunyuan-retry')

    def load(self, **kwargs):
        calls.append(('loader',))
        payload = {'model_type': PIPELINE, 'repo_id': HUNYUAN_720P_T2V_REPOSITORY, 'repo_source': 'hub',
                   'revision': HUNYUAN_720P_T2V_REVISION, 'vae': {'model_id': model_id}}
        return bind_loader_outputs({'pipeline_components': payload}, _token())

    def state(stage, kwargs, **values):
        token = require_component_binding(kwargs['pipeline_components'], label='fixture', expected_role='pipeline_components')
        from diffusers.modular_pipelines.modular_pipeline import PipelineState
        return {'state_out': _issue_workflow_state(token=token, pipeline_class=PIPELINE,
                workflow_id='text2video', completed_stage=stage, state=PipelineState(values=values))}

    def encode(self, **kwargs):
        calls.append(('encode',))
        return state('text_encoder', kwargs)

    def sample(self, **kwargs):
        calls.append(('sample', kwargs['seed']))
        return state('denoise', kwargs, latents=torch.zeros(1, 32, 2, 16, 16))

    actual_export = ExportAsset.execute

    def export(self, *, video, **kwargs):
        assert len(video) == 5 and all(isinstance(frame, Image.Image) and frame.size == (256, 256)
                                       for frame in video)
        calls.append(('export',))
        return actual_export(self, video=video, **kwargs)

    for cls, execute in ((loaders.ModelsLoader, load), (blocks.WorkflowHunyuanVideo15TextEncode, encode),
                         (blocks.WorkflowHunyuanVideo15Denoise, sample), (ExportAsset, export)):
        monkeypatch.setattr(cls, 'execute', execute)

    def app():
        server = WebServer(modules=modules.MODULE_MAP, work_dir=str(tmp_path), data_dir=str(tmp_path / 'data'))
        monkeypatch.setattr(importlib.import_module('modiff.NodeBase'), '_server', lambda: server)
        monkeypatch.setattr(importlib.import_module('modiff.server'), 'server', server)
        server.backend_source_identity = {'fingerprint': SOURCE}
        server._last_runtime_fingerprint = {'fingerprint': RUNTIME}
        monkeypatch.setattr(server, '_runtime_fingerprint', lambda **kwargs: {'fingerprint': RUNTIME})
        monkeypatch.setattr(server, '_runtime_fingerprint_for_control_request', lambda: {'fingerprint': RUNTIME})
        monkeypatch.setattr(server, 'queue_message', Mock())
        return server

    async def attempt(server, submit):
        worker = asyncio.create_task(server._main_worker())
        try:
            task_id = await submit()
            await asyncio.wait_for(server.main_queue.join(), timeout=45)
            return task_id
        finally:
            await server.main_queue.put(None)
            await worker
            if server._model_executor:
                server._model_executor.shutdown(wait=True)

    async def fail_initial_decode():
        first = app()
        task_id = await attempt(first, lambda: first.queue_task(
            first.execute_graph, (_graph(False),), None, 'original', name='Graph execution'))
        failed = first.recent_tasks[0]
        assert failed['task_id'] == task_id and failed['status'] == 'failed'
        assert failed['node'] == 'decode' and failed['oom'] is True
        assert failed['decodeRecoveryCheckpoint']
        assert first.node_cache == {}
        assert not vae.use_tiling
        assert calls == [('loader',), ('encode',), ('sample', 7), ('vae', False, (1, 32, 2, 16, 16))]
        return task_id

    async def retry_saved_decode():
        reopened = app()
        assert len(reopened.recent_tasks) == 1 and reopened.recent_tasks[0]['task_id'] == source_id
        failed = reopened.recent_tasks[0]
        assert reopened.task_graphs == {}
        assert decode_recovery_status(reopened, failed)['status'] == 'ready'
        request = SimpleNamespace(match_info={'task_id': source_id},
                                  read=lambda: asyncio.sleep(0, result=b'{"sid":"reopened","enableVaeTiling":true}'))

        async def retry():
            response = await reopened.retry_decode(request)
            assert response.status == 200, response.text
            return json.loads(response.text)['task_id']

        retry_id = await attempt(reopened, retry)
        result = reopened.recent_tasks[0]
        assert result['task_id'] == retry_id and result['status'] == 'completed'
        assert result['recoveredFromTaskId'] == source_id
        assert result['decodeRecoverySettings'] == {
            'originalVaeTiling': False, 'requestedVaeTiling': True, 'effectiveVaeTiling': True,
        }
        from modiff.media_assets import probe_video_file

        asset = reopened.node_cache['output'].output['asset']
        assert asset['task_id'] == retry_id
        media = probe_video_file(asset['path'])
        assert (media['width'], media['height'], media['frame_count']) == (256, 256, 5)
        detail = json.loads((await reopened.get_run(SimpleNamespace(match_info={'task_id': retry_id}))).text)
        assert len(detail['outputs']) == 1
        assert detail['outputs'][0]['decodeRecovery'] == {
            'sourceTaskId': source_id, 'reusedStage': 'denoise', **result['decodeRecoverySettings'],
        }
        assert decode_recovery_status(reopened, failed)['status'] == 'unavailable'
        assert reopened.recent_tasks[1] == failed
        assert not vae.use_tiling
        assert calls == [('loader',), ('vae', True, (1, 32, 2, 16, 16)), ('export',)]

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return asyncio.run(fail_initial_decode() if source_id is None else retry_saved_decode())
    finally:
        torch.set_num_threads(old_threads)


def _retry_in_fresh_process(tmp_path, source_id, original_pid):
    assert os.getpid() != original_pid
    with pytest.MonkeyPatch.context() as monkeypatch:
        _run_native_vae_worker(tmp_path, monkeypatch, source_id)


def test_real_native_vae_decode_after_failed_attempt_and_worker_reopen(tmp_path, monkeypatch):
    source_id = _run_native_vae_worker(tmp_path, monkeypatch)
    worker = multiprocessing.get_context('spawn').Process(
        target=_retry_in_fresh_process, args=(tmp_path, source_id, os.getpid()))
    worker.start()
    try:
        worker.join(timeout=90)
        assert worker.exitcode == 0, 'The fresh process did not complete the decode-only retry.'
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        worker.close()
