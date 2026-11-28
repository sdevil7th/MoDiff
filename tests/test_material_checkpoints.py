"""Retained tensors survive runtime cache release without serializing owners."""
from types import SimpleNamespace
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file

from modiff import material_checkpoints as checkpoints
from modiff.hunyuan_video15_artifacts import HUNYUAN_720P_T2V_REPOSITORY, HUNYUAN_720P_T2V_REVISION
from modules.ModularDiffusers.route_state import issue_pipeline_instance_token
from modules.ModularDiffusers.workflow_blocks import WorkflowHunyuanVideo15Denoise, _issue_workflow_state


def completed(tmp_path, *, dtype=torch.bfloat16, stage='denoise'):
    node = WorkflowHunyuanVideo15Denoise('sampled-stage')
    token = issue_pipeline_instance_token(model_type='HunyuanVideo15ModularPipeline',
        repo_id=HUNYUAN_720P_T2V_REPOSITORY, repo_source='hub', revision=HUNYUAN_720P_T2V_REVISION)
    tensor = torch.arange(32 * 2 * 4 * 4).reshape(1, 32, 2, 4, 4).to(dtype)
    state = _issue_workflow_state(token=token, pipeline_class='HunyuanVideo15ModularPipeline',
        workflow_id='text2video', completed_stage=stage, state={'latents': tensor})
    node._has_changed = True
    node.output = {'state_out': state}
    node.record_generation_inputs({'seed': 17, 'num_inference_steps': 50})
    server = SimpleNamespace(node_cache={node.node_id: node}, data_dir=tmp_path,
        current_task={'task_id': 'task-completed-sampling', 'runtimeFingerprint': 'sha256:' + '7' * 64,
                      'runtimeHints': {'runInputHash': '8' * 64}},
        backend_source_identity={'fingerprint': 'sha256:' + '9' * 64})
    return server, node, tensor


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_completed_native_material_remains_exact_after_cache_cleanup(tmp_path, dtype):
    server, node, original = completed(tmp_path, dtype=dtype)
    expected = original.clone()
    record = checkpoints.capture_completed_material(server, node.node_id)
    original.add_(1)
    server.node_cache.clear()  # Failed-attempt teardown may release every owner.
    root = tmp_path / 'runtime/material-checkpoints'
    metadata_path = next(root.glob('*.json'))
    assert json.loads(metadata_path.read_text()) == record
    body = (root / record['tensorFile']).read_bytes()
    assert record['tensorSHA256'] == hashlib.sha256(body).hexdigest()
    loaded = load_file(str(root / record['tensorFile']))
    assert list(loaded) == ['latents']
    assert torch.equal(loaded['latents'], expected)
    assert loaded['latents'].device.type == 'cpu' and not loaded['latents'].requires_grad
    assert record['restorationAuthorized'] is False
    assert record['sourceTaskId'] == 'task-completed-sampling'
    assert record['runInputHash'] == '8' * 64
    assert record['revision'] == HUNYUAN_720P_T2V_REVISION
    assert 'pipeline_token' not in metadata_path.read_text()


@pytest.mark.parametrize('case', ['cache_reuse', 'no_task', 'unsupported_dtype', 'oversized', 'metadata_bound'])
def test_ineligible_material_is_not_copied_or_saved(tmp_path, monkeypatch, case):
    server, node, _ = completed(tmp_path, dtype=torch.int64 if case == 'unsupported_dtype' else torch.bfloat16)
    if case == 'cache_reuse':
        node._has_changed = False
    elif case == 'no_task':
        server.current_task = None
    elif case == 'oversized':
        monkeypatch.setattr(checkpoints, 'MAX_TENSOR_BYTES', 1)
    elif case == 'metadata_bound':
        node._execution_input_record = {'oversized': 'x' * checkpoints.MAX_METADATA_BYTES}
    assert checkpoints.capture_completed_material(server, node.node_id) is None
    assert not list(tmp_path.rglob('*.safetensors'))


def test_unissued_or_wrong_stage_cannot_become_checkpoint_authority(tmp_path):
    server, node, _ = completed(tmp_path, stage='text_encoder')
    with pytest.raises(ValueError, match='required stage'):
        checkpoints.capture_completed_material(server, node.node_id)
    assert not (tmp_path / 'runtime').exists()


def test_checkpoint_parent_escape_is_rejected_before_write(tmp_path):
    server, node, _ = completed(tmp_path / 'private')
    server.data_dir.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    try:
        (server.data_dir / 'runtime').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('Host does not permit directory symlink creation')
    with pytest.raises(ValueError, match='boundary'):
        checkpoints.capture_completed_material(server, node.node_id)
    assert not list(outside.iterdir())


def test_other_nodes_are_not_inspected_as_model_outputs(tmp_path):
    class Other:
        @property
        def output(self):
            raise AssertionError('Do not inspect custom output suppliers.')
    server = SimpleNamespace(node_cache={'custom': Other()}, data_dir=tmp_path)
    assert checkpoints.capture_completed_material(server, 'custom') is None


def test_ordinary_dispatch_captures_success_before_a_downstream_decode_fault(tmp_path, monkeypatch):
    """Synthetic CPU stage result through real dispatch; no model-run claim."""
    from collections import Counter
    import importlib
    from unittest.mock import Mock
    import modules
    from modiff.NodeBase import memory_manager
    from modiff.server import WebServer
    from modules.ModularDiffusers.workflow_blocks import WorkflowHunyuanVideo15Decode

    seed_server, _, tensor = completed(tmp_path)
    sampler_state = seed_server.node_cache['sampled-stage'].output['state_out']
    def sample(self, **kwargs):
        return {'state_out': sampler_state}
    def decode(self, **kwargs):
        assert list((tmp_path / 'runtime/material-checkpoints').glob('*.json'))
        raise ValueError('Synthetic decoder fault after completed sampling')
    monkeypatch.setattr(WorkflowHunyuanVideo15Denoise, 'execute', sample)
    monkeypatch.setattr(WorkflowHunyuanVideo15Decode, 'execute', decode)
    monkeypatch.setattr(memory_manager, 'cache', {})
    monkeypatch.setattr(memory_manager, '_active_ids', Counter())
    graph = {'sid': None, 'nodes': {
        'sampled-stage': {'module': 'modules.ModularDiffusers', 'action': 'WorkflowHunyuanVideo15Denoise', 'params': {}},
        'decode-stage': {'module': 'modules.ModularDiffusers', 'action': 'WorkflowHunyuanVideo15Decode',
                         'params': {'state_in': {'sourceId': 'sampled-stage', 'sourceKey': 'state_out'}}},
    }, 'paths': [['sampled-stage', 'decode-stage']]}
    app = object.__new__(WebServer)
    app.modules, app.node_cache, app.data_dir = modules.MODULE_MAP, {}, tmp_path
    app.instance = 'explicit-checkpoint-cpu-unit-worker'
    app.backend_source_identity = seed_server.backend_source_identity
    app.current_task = {**seed_server.current_task, 'args': [graph], 'sid': None,
                        'attempt_index': 0, 'current_node': 'sampled-stage', 'interrupt_requested': False}
    app._active_graph_node_ids = set(graph['nodes'])
    app.queue_message = Mock()
    monkeypatch.setattr(importlib.import_module('modiff.NodeBase'), '_server', lambda: app)
    app.execute_node('sampled-stage', graph['nodes']['sampled-stage'], None, quiet=True)
    app.current_task['current_node'] = 'decode-stage'
    with pytest.raises(RuntimeError, match='Synthetic decoder fault'):
        app.execute_node('decode-stage', graph['nodes']['decode-stage'], None, quiet=True)
    app.node_cache.clear()
    root = tmp_path / 'runtime/material-checkpoints'
    record = json.loads(next(root.glob('*.json')).read_text())
    assert torch.equal(load_file(str(root / record['tensorFile']))['latents'], tensor)
