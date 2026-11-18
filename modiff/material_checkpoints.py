"""Bounded material retention after trusted native stages.

Only tensor bytes and inspected receipt metadata are saved. Pipeline State,
models and process-local binding tokens are never serialized. Recovery must
separately validate the original task, graph and runtime before issuing new state.
"""
from pathlib import Path
import hashlib
import hmac
import json
import os
import re
import tempfile
import threading
import time

MAX_TENSOR_BYTES = 512 * 1024 * 1024
MAX_TENSOR_FILE_BYTES = MAX_TENSOR_BYTES + 65536
MAX_METADATA_BYTES = 262144
MAX_STORED_BYTES = 2 * 1024 * 1024 * 1024
MAX_STORED_PAIRS = 8
RETENTION_SECONDS = 48 * 60 * 60
CHECKPOINT_SUBDIRECTORY = Path('runtime/material-checkpoints')
_LOCK = threading.RLock()


def _directory(data_dir, *, create=False):
    data_root = Path(data_dir).resolve()
    root = data_root / CHECKPOINT_SUBDIRECTORY
    if root.resolve() != root or not root.is_relative_to(data_root):
        raise ValueError('Material checkpoint directory escapes its data boundary.')
    if create:
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
    return root


def is_private_material_path(path, data_dir):
    root = Path(data_dir).resolve() / CHECKPOINT_SUBDIRECTORY
    return Path(path).resolve().is_relative_to(root.resolve())


def _exact_bytes(path, maximum):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Material checkpoint file is missing or unsafe.')
    with path.open('rb') as stream:
        content = stream.read(maximum + 1)
    if not 0 < len(content) <= maximum:
        raise ValueError('Material checkpoint file exceeds its size bounds.')
    return content


def _key(task_id, node_id):
    return hashlib.sha256(json.dumps([task_id, node_id], separators=(',', ':')).encode()).hexdigest()


def _prune(root, *, incoming_bytes=0, replace_key=None):
    """Remove only application-owned pairs and abandoned temporary files."""
    if not root.is_dir():
        return
    now = time.time()
    candidates = {}
    for file in root.iterdir():
        if re.fullmatch(r'\.checkpoint-[a-z0-9_]+\.tmp', file.name):
            if file.is_file() or file.is_symlink():
                file.unlink(missing_ok=True)
        elif re.fullmatch(r'[a-f0-9]{64}\.(json|safetensors)', file.name):
            candidates.setdefault(file.stem, {})[file.suffix] = file

    def remove(pair):
        for file in pair.values():
            if file.is_file() or file.is_symlink():
                file.unlink(missing_ok=True)

    retained = []
    for key, pair in candidates.items():
        if key == replace_key or len(pair) != 2 or any(
                file.is_symlink() or not file.is_file() for file in pair.values()):
            remove(pair)
            continue
        metadata, tensor = pair['.json'].stat(), pair['.safetensors'].stat()
        if (not 0 < metadata.st_size <= MAX_METADATA_BYTES
                or not 0 < tensor.st_size <= MAX_TENSOR_FILE_BYTES
                or not 0 <= now - metadata.st_mtime < RETENTION_SECONDS):
            remove(pair)
            continue
        retained.append((metadata.st_mtime, key, metadata.st_size + tensor.st_size, pair))

    total = sum(item[2] for item in retained)
    count = len(retained)
    for _, _, size, pair in sorted(retained):
        if total + incoming_bytes <= MAX_STORED_BYTES and count + bool(incoming_bytes) <= MAX_STORED_PAIRS:
            break
        remove(pair)
        count -= 1
        total -= size
    if total + incoming_bytes > MAX_STORED_BYTES or count + bool(incoming_bytes) > MAX_STORED_PAIRS:
        raise ValueError('Material checkpoint exceeds the application retention budget.')


def prune_material_checkpoints(data_dir):
    with _LOCK:
        _prune(_directory(data_dir))


def _digest(path):
    hasher = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(block)
            if size > MAX_TENSOR_FILE_BYTES:
                raise ValueError('Material checkpoint file exceeds its size bounds.')
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
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_material_checkpoint(data_dir, *, task_id, source_node_id, receipt_sha256, load_tensor=False,
                             verify_tensor=False):
    """Verify exact bytes against the originating server's terminal-history digest."""
    if not isinstance(receipt_sha256, str) or not re.fullmatch(r'[a-f0-9]{64}', receipt_sha256):
        raise ValueError('The originating run did not authorize a material checkpoint.')
    key = _key(task_id, source_node_id)
    with _LOCK:
        root = _directory(data_dir)
        metadata = root / (key + '.json')
        payload = _exact_bytes(metadata, MAX_METADATA_BYTES)
        if (not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), receipt_sha256)
                or not 0 <= time.time() - metadata.stat().st_mtime < RETENTION_SECONDS):
            raise ValueError('The retained sampling receipt is stale or no longer matches the original run.')
        record = json.loads(payload)
        name = key + '.safetensors'
        if (not isinstance(record, dict) or record.get('schemaVersion') != 2
                or record.get('sourceTaskId') != task_id or record.get('sourceNodeId') != source_node_id
                or record.get('tensorFile') != name or not isinstance(record.get('recovery'), dict)):
            raise ValueError('The retained sampling receipt does not belong to this run.')
        tensor_path = root / name
        if tensor_path.is_symlink() or not tensor_path.is_file() or tensor_path.stat().st_size != record.get('tensorFileBytes'):
            raise ValueError('The retained tensor is missing or no longer matches its receipt.')
        if not load_tensor:
            if verify_tensor and not hmac.compare_digest(_digest(tensor_path), record['tensorSHA256']):
                raise ValueError('The retained tensor no longer matches its receipt.')
            return record
        body = _exact_bytes(tensor_path, MAX_TENSOR_FILE_BYTES)
        if not hmac.compare_digest(hashlib.sha256(body).hexdigest(), record['tensorSHA256']):
            raise ValueError('The retained tensor no longer matches its receipt.')

    import torch
    from safetensors import SafetensorError
    from safetensors.torch import load
    try:
        tensors = load(body)
    except SafetensorError as exc:
        raise ValueError('The retained tensor file is invalid.') from exc
    tensor = tensors.get('latents')
    expected = record['tensor']
    if (set(tensors) != {'latents'} or not isinstance(tensor, torch.Tensor)
            or list(tensor.shape) != expected['shape'] or str(tensor.dtype) != expected['dtype']
            or tensor.numel() * tensor.element_size() != expected['bytes']
            or not torch.isfinite(tensor).all().item()):
        raise ValueError('The tensor contents do not match the retained sampling geometry and dtype.')
    return record, tensor


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
    if type(node) is not WorkflowHunyuanVideo15Denoise:
        return None
    task = getattr(server, 'current_task', None)
    if not isinstance(task, dict) or not isinstance(task.get('task_id'), str) or not task['task_id']:
        return None
    if not node._has_changed:
        task['_decode_recovery_blocker'] = (
            'This run reused in-memory sampling from another run. No durable sampling was captured for this attempt.'
        )
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
    key = _key(task['task_id'], node_id)
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
    from modiff.material_decode_recovery import describe_completed_sampling
    recovery = describe_completed_sampling(server, task, record, state)
    if recovery is not None:
        record.update(schemaVersion=2, recovery=recovery)
        if len(json.dumps(record, allow_nan=False).encode()) > MAX_METADATA_BYTES - 256:
            record.pop('recovery')
            record['schemaVersion'] = 1
    # Validate receipt serializability/bounds before copying accelerator memory.
    if len(json.dumps(record, allow_nan=False).encode()) > MAX_METADATA_BYTES - 256:
        return None
    from safetensors.torch import save_file
    with _LOCK:
        root = _directory(server.data_dir, create=True)
        _prune(root, incoming_bytes=size + 65536 + MAX_METADATA_BYTES, replace_key=key)
        retained = tensor.detach().to(device='cpu', copy=True).contiguous()
        tensor_path = _publish(root, record['tensorFile'], lambda path: save_file({'latents': retained}, str(path)))
        try:
            record['tensorSHA256'] = _digest(tensor_path)
            if record['schemaVersion'] == 2:
                record['tensorFileBytes'] = tensor_path.stat().st_size
            encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
            if len(encoded) > MAX_METADATA_BYTES:
                raise ValueError('Material checkpoint metadata exceeds its finite limit.')
            _publish(root, key + '.json', lambda path: path.write_bytes(encoded))
        except BaseException:
            tensor_path.unlink(missing_ok=True)
            raise
    if record['schemaVersion'] == 2:
        task['_material_checkpoint'] = {
            'sourceNodeId': node_id, 'decodeNodeId': recovery['decodeNodeId'],
            'receiptSHA256': hashlib.sha256(encoded).hexdigest(), 'attemptIndex': task.get('attempt_index', 0),
        }
    return record
