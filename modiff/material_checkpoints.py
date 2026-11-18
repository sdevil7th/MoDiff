"""Bounded material retention after trusted native stages, not restore authority.

Only tensor bytes and inspected receipt metadata are saved. Pipeline State,
models and process-local binding tokens are never serialized. Recovery must
separately validate the original task, graph and runtime before issuing new state.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile

MAX_TENSOR_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 262144


def _digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _publish(root, name, write):
    target = root / name
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError('Material checkpoint target escapes its private directory.')
    descriptor, temporary_name = tempfile.mkstemp(dir=root, prefix='.checkpoint-', suffix='.tmp')
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def capture_completed_material(server, node_id):
    """Retain an eligible builtin's completed denoise result before decode.

    Called by ordinary dispatch after success, under its existing cache/model
    lease. Returning None is an unsupported capture, never an execution failure.
    No custom supplier, constructor, pickle or tensor is inspected on discovery.
    """
    node = server.node_cache.get(node_id)
    if (type(node).__module__, type(node).__name__) != (
            'modules.ModularDiffusers.workflow_blocks', 'WorkflowHunyuanVideo15Denoise'):
        return None
    from modules.ModularDiffusers.workflow_blocks import WorkflowHunyuanVideo15Denoise, _require_workflow_state
    if type(node) is not WorkflowHunyuanVideo15Denoise or not node._has_changed:
        return None
    task = getattr(server, 'current_task', None)
    if not isinstance(task, dict) or not isinstance(task.get('task_id'), str) or not task['task_id']:
        return None
    value = node.output.get('state_out')
    token = getattr(value, '_pipeline_token', None)
    state = _require_workflow_state(value, token=token, pipeline_class='HunyuanVideo15ModularPipeline',
        workflow_id=getattr(value, '_workflow_id', None), completed_stage='denoise')
    repository, revision = getattr(token, '_repo_id', None), getattr(token, '_revision', None)
    if getattr(token, '_repo_source', None) != 'hub' or not isinstance(repository, str) or not re.fullmatch(
            r'[a-f0-9]{40}', revision or ''):
        return None
    import torch
    tensor = state.get('latents')
    if type(tensor) is not torch.Tensor or tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return None
    size = tensor.numel() * tensor.element_size()
    if not 0 < size <= MAX_TENSOR_BYTES or tensor.ndim != 5:
        return None
    data_root = Path(server.data_dir).resolve()
    root = (data_root / 'runtime/material-checkpoints').resolve()
    if not root.is_relative_to(data_root):
        raise ValueError('Material checkpoint directory escapes its data boundary.')
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([task['task_id'], node_id], separators=(',', ':')).encode()).hexdigest()
    record = {
        'schemaVersion': 1, 'restorationAuthorized': False,
        'sourceTaskId': task['task_id'], 'sourceNodeId': node_id,
        'workflowId': value._workflow_id, 'completedStage': 'denoise',
        'modelType': value._pipeline_class, 'repository': repository, 'revision': revision,
        'runtimeFingerprint': task.get('runtimeFingerprint'),
        'runInputHash': (task.get('runtimeHints') or {}).get('runInputHash') or task.get('run_input_hash'),
        'backendSource': getattr(server, 'backend_source_identity', {}).get('fingerprint'),
        'generationInputs': getattr(node, '_execution_input_record', None),
        'tensor': {'name': 'latents', 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                   'sourceDevice': str(tensor.device), 'bytes': size},
        'tensorFile': key + '.safetensors',
    }
    # Validate receipt serializability/bounds before copying accelerator memory.
    if len(json.dumps(record, allow_nan=False).encode()) > MAX_METADATA_BYTES - 256:
        return None
    from safetensors.torch import save_file
    retained = tensor.detach().to(device='cpu', copy=True).contiguous()
    tensor_path = _publish(root, record['tensorFile'], lambda path: save_file({'latents': retained}, str(path)))
    record['tensorSHA256'] = _digest(tensor_path)
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2).encode()
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError('Material checkpoint metadata exceeds its finite limit.')
    _publish(root, key + '.json', lambda path: path.write_bytes(encoded))
    return record
