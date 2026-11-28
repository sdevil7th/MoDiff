"""Full-byte verification, task-bound progress and cooperative Stop controls."""
import hashlib

import pytest

from modiff.controlled_artifacts import _sha256_file

CHUNK = 8 * 1024 * 1024


def test_interrupted_hash_read_preserves_cancellation_type(tmp_path):
    path = tmp_path / 'real-multiple-chunk.bin'
    path.write_bytes(b'a' * (CHUNK + 17))
    reads = []

    class InterruptedPath:
        def stat(self):
            return path.stat()

        def open(self, mode):
            handle = path.open(mode)

            class Reader:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    handle.close()

                def read(self, count):
                    reads.append(count)
                    if len(reads) == 2:
                        raise InterruptedError('Stop during full-byte verification')
                    return handle.read(count)

            return Reader()

    with pytest.raises(InterruptedError, match='Stop during full-byte'):
        _sha256_file(InterruptedPath())
    assert reads == [CHUNK, CHUNK]


# These controls use real local reads. Tiny SDK cases reuse the existing
# explicitly synthetic artifact fixture, never publisher/model qualification.
from copy import deepcopy
import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock

from modiff import artifact_verification as verification
from modiff import controlled_artifacts, modular_weight_formats as formats
from modiff.native_component_lora import native_component_graph_scope
import test_modular_weight_formats as weight_format_tests

PAIRS = weight_format_tests.PAIRS
runtime = weight_format_tests.runtime


def owned(monkeypatch, sid='unit-session', node=None):
    source = {'module': 'modules.ModularDiffusers', 'action': 'ModelsLoader', 'params': {}}
    graph = {'sid': sid, 'nodes': {'models': source}}
    task = {'task_id': 'unit-task', 'attempt_index': 2, 'sid': sid, 'args': [graph],
            'current_node': 'models', 'interrupt_requested': False,
            'runtimeHints': {'clientRunId': 'unit-run', 'runInputHash': 'unit-hash'}}
    if node is None:
        node = SimpleNamespace(node_id='models', _sid=sid, _interrupt=False, progress=Mock())
    else:
        node._sid = sid
    server = SimpleNamespace(current_task=task, node_cache={'models': node},
        _current_run_identity_payload=lambda: {'client_run_id': 'unit-run', 'run_input_hash': 'unit-hash'},
        record_node_progress=Mock(side_effect=lambda value: value), queue_message=Mock())
    monkeypatch.setattr(importlib.import_module('modiff.NodeBase'), '_server', lambda: server)
    return SimpleNamespace(graph=graph, task=task, node=node, server=server)


def test_real_multi_chunk_digest_and_all_byte_witnesses(tmp_path):
    path = tmp_path / 'multi.bin'
    raw = b'abcd' * (CHUNK // 2) + b'last-17-bytes!!!!'
    path.write_bytes(raw)
    calls = []
    assert _sha256_file(path, _verification_callback=lambda *value: calls.append(value)) == hashlib.sha256(raw).hexdigest()
    size = len(raw)
    assert calls == [(0, size, False), (CHUNK, size, False),
                     (CHUNK * 2, size, False), (size, size, False), (size, size, True)]


@pytest.mark.parametrize('when', ['before', 'middle', 'after'])
def test_cooperative_cancellation_at_every_hash_boundary(tmp_path, when):
    path = tmp_path / 'multi.bin'
    path.write_bytes(b'a' * (2 * CHUNK + 3))
    calls = []
    def check(completed, total, finished):
        calls.append((completed, finished))
        if ((when == 'before' and completed == 0) or
                (when == 'middle' and completed == CHUNK) or (when == 'after' and finished)):
            raise InterruptedError('owned Stop')
    with pytest.raises(InterruptedError, match='owned Stop'):
        _sha256_file(path, _verification_callback=check)
    assert calls[-1] == {'before': (0, False), 'middle': (CHUNK, False), 'after': (2 * CHUNK + 3, True)}[when]


def test_real_in_place_mutation_remains_refused(tmp_path):
    path = tmp_path / 'multi.bin'
    path.write_bytes(b'a' * (CHUNK + 3))
    original = path.stat()
    def mutate(completed, total, finished):
        if completed == CHUNK:
            with path.open('r+b') as handle:
                handle.seek(-1, 2)
                handle.write(b'z')
                handle.flush()
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match='changed while'):
        _sha256_file(path, _verification_callback=mutate)


def test_task_progress_is_indeterminate_throttled_and_never_claims_verified(monkeypatch):
    context = owned(monkeypatch)
    with native_component_graph_scope(context.graph, 'unit-source'):
        observer = verification.artifact_verification_for_loader(context.node, 'fp8r')
        clock = iter([observer._started, observer._started + .1, observer._started + .26, observer._started + .27])
        observer._clock = lambda: next(clock)
        for done, finished in [(0, False), (1, False), (2, False), (3, True)]:
            observer('weight.safetensors', done, 3, finished)
    calls = context.node.progress.call_args_list
    assert len(calls) == 3
    assert all(call.args == (-1,) and call.kwargs['phase'] == 'loading' for call in calls)
    assert all(call.kwargs['message'].startswith('Verifying weight.safetensors:') for call in calls)
    assert [call.kwargs['message'].split(': ')[1] for call in calls] == ['0/3 bytes', '2/3 bytes', '3/3 bytes']


def test_stop_is_checked_even_when_publication_is_throttled(monkeypatch):
    context = owned(monkeypatch)
    with native_component_graph_scope(context.graph, 'unit-source'):
        observer = verification.artifact_verification_for_loader(context.node, 'fp8r')
        observer._clock = lambda: observer._started
        observer('weight', 0, 3, False)
        context.task['interrupt_requested'] = True
        with pytest.raises(InterruptedError, match='current task'):
            observer('weight', 1, 3, False)
    assert context.node.progress.call_count == 1


def test_preconstruction_progress_retains_exact_run_task_and_attempt(monkeypatch):
    context = owned(monkeypatch)
    with native_component_graph_scope(context.graph, 'unit-source'):
        observer = verification.artifact_verification_for_current_node(context.server, 'models', 'unit-session')
        observer('weight', 0, 7, False)
    payload = context.server.queue_message.call_args.args[0]
    assert {key: payload[key] for key in ('node', 'task_id', 'attempt_index', 'client_run_id', 'run_input_hash')} == {
        'node': 'models', 'task_id': 'unit-task', 'attempt_index': 2,
        'client_run_id': 'unit-run', 'run_input_hash': 'unit-hash'}
    assert payload['progress'] == -1 and payload['status'] == 'running'
    assert context.node.progress.call_count == 0


@pytest.mark.parametrize('fault', ['scope', 'graph', 'task', 'sid', 'current', 'node', 'action'])
def test_unrelated_or_stale_authority_gets_no_observer(monkeypatch, fault):
    context = owned(monkeypatch)
    graph = context.graph
    if fault == 'graph':
        context.task['args'] = [deepcopy(graph)]
    elif fault == 'task':
        context.server.current_task = None
    elif fault == 'sid':
        context.node._sid = 'previous-session'
    elif fault == 'current':
        context.task['current_node'] = 'other'
    elif fault == 'node':
        context.server.node_cache['models'] = object()
    elif fault == 'action':
        graph['nodes']['models']['action'] = 'Other'
    with native_component_graph_scope(deepcopy(graph) if fault == 'scope' else graph, 'unit-source'):
        assert verification.artifact_verification_for_loader(context.node, 'fp8r') is None
    assert not context.node.progress.called and not context.server.queue_message.called


@pytest.mark.parametrize('fault', ['task', 'graph', 'attempt', 'sid', 'node', 'current'])
def test_captured_authority_cannot_follow_a_replacement(monkeypatch, fault):
    context = owned(monkeypatch)
    with native_component_graph_scope(context.graph, 'unit-source'):
        observer = verification.artifact_verification_for_loader(context.node, 'fp8r')
        if fault == 'task':
            context.server.current_task = dict(context.task)
        elif fault == 'graph':
            context.task['args'] = [deepcopy(context.graph)]
        elif fault == 'attempt':
            context.task['attempt_index'] += 1
        elif fault == 'sid':
            context.node._sid = 'new-session'
        elif fault == 'node':
            context.server.node_cache['models'] = object()
        else:
            context.task['current_node'] = 'other'
        with pytest.raises(InterruptedError, match='no longer belongs'):
            observer('weight', 0, 3, False)
    assert context.task['interrupt_requested'] is False
    assert not context.node.progress.called


def test_api_task_without_websocket_sid_still_honors_stop_without_events(monkeypatch):
    context = owned(monkeypatch, sid=None)
    with native_component_graph_scope(context.graph, 'unit-source'):
        observer = verification.artifact_verification_for_loader(context.node, 'nvfp4')
        assert observer is not None
        observer('weight', 0, 3, False)
        context.task['interrupt_requested'] = True
        with pytest.raises(InterruptedError, match='current task'):
            observer('weight', 1, 3, False)
    assert not context.node.progress.called and not context.server.queue_message.called


def test_old_node_interrupt_does_not_cancel_new_owned_task(monkeypatch):
    context = owned(monkeypatch)
    context.node._interrupt = True
    with native_component_graph_scope(context.graph, 'unit-source'):
        observer = verification.artifact_verification_for_loader(context.node, 'fp8r')
        observer('weight', 0, 3, False)
    assert context.node.progress.call_count == 1
    assert verification.artifact_verification_for_loader(context.node, 'native') is None
    assert formats.resolve_weight_format(model_type='old', metadata={}, repository='old/repo', revision=None,
                                         _verification_callback=Mock(side_effect=AssertionError('native event'))) is None


def actual_node(runtime, monkeypatch):
    node = runtime.loaders.ModelsLoader('models')
    runtime.nodes.append(node)
    context = owned(monkeypatch, node=node)
    return node, context


def test_actual_cold_and_warm_loader_rehashes_all_bytes_and_publishes(runtime, monkeypatch):
    node, context = actual_node(runtime, monkeypatch)
    real = controlled_artifacts._sha256_file
    hashed = []
    def tracked(path, **kwargs):
        hashed.append((str(path), kwargs.get('_verification_callback') is not None))
        return real(path, **kwargs)
    monkeypatch.setattr(controlled_artifacts, '_sha256_file', tracked)
    with native_component_graph_scope(context.graph, 'unit-source'):
        runtime.call(node=node)
        owner = node.loader
        assert len(hashed) == 4 and all(bound for _, bound in hashed)
        node_count = len(runtime.manager.components)
        runtime.call(node=node)
    assert node.loader is owner and node._cache_valid
    assert len(hashed) == 6 and all(bound for _, bound in hashed)
    assert len(runtime.manager.components) == node_count
    assert context.server.record_node_progress.called


@pytest.mark.parametrize('when', ['before', 'middle', 'after'])
def test_actual_loader_cancel_before_sdk_or_cache_mutation(runtime, monkeypatch, when):
    node, context = actual_node(runtime, monkeypatch)
    original_params, original_output = deepcopy(node.params), dict(node.output)
    real = controlled_artifacts._sha256_file
    callbacks = []
    def intercepted(path, **kwargs):
        original = kwargs['_verification_callback']
        def checked(done, total, finished):
            callbacks.append((done, finished))
            if ((when == 'before' and done == 0) or (when == 'middle' and done > 0 and not finished)
                    or (when == 'after' and finished)):
                context.task['interrupt_requested'] = True
            original(done, total, finished)
        return real(path, _verification_callback=checked)
    monkeypatch.setattr(controlled_artifacts, '_sha256_file', intercepted)
    from diffusers import ComponentSpec
    sdk = Mock(side_effect=AssertionError('SDK must not run after Stop'))
    monkeypatch.setattr(ComponentSpec, 'load', sdk)
    with native_component_graph_scope(context.graph, 'unit-source'), pytest.raises(InterruptedError, match='current task'):
        runtime.call(node=node)
    assert callbacks and not sdk.called and not runtime.manager.collections
    assert node.params == original_params and node.output == original_output and node.loader is None


def test_stop_between_preflight_and_nodebase_reset_refuses_before_sdk(runtime, monkeypatch):
    node, context = actual_node(runtime, monkeypatch)
    base = importlib.import_module('modiff.NodeBase').NodeBase
    original = base.__call__
    def stop_before_reset(self, **kwargs):
        context.task['interrupt_requested'] = True
        self._interrupt = True
        return original(self, **kwargs)
    monkeypatch.setattr(base, '__call__', stop_before_reset)
    from diffusers import ComponentSpec
    sdk = Mock(side_effect=AssertionError('SDK must not run after Stop'))
    monkeypatch.setattr(ComponentSpec, 'load', sdk)
    with native_component_graph_scope(context.graph, 'unit-source'), pytest.raises(RuntimeError) as error:
        runtime.call(node=node)
    assert isinstance(error.value.__cause__, InterruptedError)
    assert node._interrupt is False  # NodeBase reset cannot erase persistent task Stop.
    assert not sdk.called and not runtime.manager.collections and node.loader is None


def test_warm_cache_stop_before_return_cannot_report_success(runtime, monkeypatch):
    node, context = actual_node(runtime, monkeypatch)
    with native_component_graph_scope(context.graph, 'unit-source'):
        runtime.call(node=node)
        owner = node.loader
        base = importlib.import_module('modiff.NodeBase').NodeBase
        original = base.__call__
        def stop_after_cache(self, **kwargs):
            result = original(self, **kwargs)
            assert self.loader is owner and self._cache_valid
            context.task['interrupt_requested'] = True
            return result
        monkeypatch.setattr(base, '__call__', stop_after_cache)
        with pytest.raises(InterruptedError, match='current task'):
            runtime.call(node=node)
    assert node.loader is owner


def test_bad_digest_never_publishes_verification_success(runtime, monkeypatch):
    node, context = actual_node(runtime, monkeypatch)
    row = next(row for row in runtime.recipes if row['modelType'] == PAIRS[0][0] and row['weightFormat'] == 'fp8r')
    path = runtime.snapshots[PAIRS[0][0]] / row['weight']['path']
    with path.open('r+b') as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    with native_component_graph_scope(context.graph, 'unit-source'), pytest.raises(ValueError, match='size/SHA-256'):
        runtime.call(node=node)
    messages = [call.args[0]['message'] for call in context.server.record_node_progress.call_args_list]
    assert messages and all(message.startswith('Verifying ') for message in messages)
    assert not runtime.manager.collections


def test_size_refusal_precedes_hash_even_with_progress(runtime, monkeypatch):
    node, context = actual_node(runtime, monkeypatch)
    path = runtime.snapshots[PAIRS[0][0]] / 'variants/fp8r/model.safetensors'
    with path.open('ab') as handle:
        handle.write(b'x')
    hashed = Mock(side_effect=AssertionError('wrong-size file must not be hashed'))
    monkeypatch.setattr(controlled_artifacts, '_sha256_file', hashed)
    with native_component_graph_scope(context.graph, 'unit-source'), pytest.raises(ValueError, match='size/SHA-256'):
        runtime.call(node=node)
    assert not hashed.called and not runtime.manager.collections


@pytest.mark.parametrize('boundary', ['graph-preflight', 'resolved-submission'])
def test_actual_selected_graph_stop_precedes_destructive_preparation(runtime, monkeypatch, boundary):
    from modiff.native_component_lora import current_component_lora_context
    from modiff.server import WebServer
    from test_flux3_bfl_lora import graph as selected_graph
    context = owned(monkeypatch)
    graph = selected_graph(PAIRS[0], 'fp8r')
    context.graph = graph
    context.task.update(args=[graph], sid=graph['sid'])
    context.node._sid = graph['sid']
    context.server.backend_source_identity = {'fingerprint': 'unit-source'}
    context.server._execute_graph_inner = Mock(side_effect=AssertionError('No attempt cleanup after Stop'))
    def stop_on_progress(payload):
        context.task['interrupt_requested'] = True
        return payload
    context.server.record_node_progress.side_effect = stop_on_progress
    from diffusers import ComponentSpec
    sdk = Mock(side_effect=AssertionError('No SDK call after Stop'))
    monkeypatch.setattr(ComponentSpec, 'load', sdk)
    with pytest.raises(InterruptedError, match='current task'):
        if boundary == 'graph-preflight':
            WebServer._execute_graph(context.server, graph)
        else:
            with native_component_graph_scope(graph, 'unit-source'):
                authority = current_component_lora_context('models')
                assert authority is not None and authority.weight_format_contract is not None
                observer = verification.artifact_verification_for_current_node(context.server, 'models', graph['sid'])
                values = {key: item['value'] for key, item in graph['nodes']['models']['params'].items()}
                authority.validate_source_submission(values, _verification_callback=observer)
    assert context.server.record_node_progress.called
    assert not context.server._execute_graph_inner.called and not sdk.called
    assert not runtime.manager.collections
