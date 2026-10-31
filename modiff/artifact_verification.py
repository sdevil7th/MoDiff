"""Task-scoped progress and cooperative cancellation for artifact verification."""
import time
from typing import Protocol


class VerificationCallback(Protocol):
    def check(self) -> None: ...

    def __call__(self, name: str, completed: int, total: int, finished: bool) -> None: ...


class _TaskArtifactVerification:
    def __init__(self, server, task, graph, node_id, sid, *, node, current_node):
        self._server, self._task, self._graph = server, task, graph
        self._node_id, self._sid, self._node = node_id, sid, node
        self._task_id, self._attempt = task.get('task_id'), task.get('attempt_index')
        self._source_node = graph['nodes'][node_id]
        self._current_node = current_node
        self._clock = time.monotonic
        self._started = self._clock()
        self._last = None

    def check(self):
        task = getattr(self._server, 'current_task', None)
        args = task.get('args') if isinstance(task, dict) else None
        nodes = self._graph.get('nodes')
        if (task is not self._task or not isinstance(args, (list, tuple)) or not args or args[0] is not self._graph
                or task.get('task_id') != self._task_id or task.get('attempt_index') != self._attempt
                or task.get('sid') != self._sid or self._graph.get('sid') != self._sid
                or not isinstance(nodes, dict) or nodes.get(self._node_id) is not self._source_node
                or (self._source_node.get('module'), self._source_node.get('action')) != ('modules.ModularDiffusers', 'ModelsLoader')
                or (self._current_node and task.get('current_node') != self._node_id)
                or (self._node is not None and (self._node._sid != self._sid
                    or self._server.node_cache.get(self._node_id) is not self._node))):
            raise InterruptedError('Artifact verification no longer belongs to the current task and node.')
        if task.get('interrupt_requested'):
            raise InterruptedError('Artifact verification interrupted by the current task.')

    def __call__(self, name: str, completed: int, total: int, finished: bool) -> None:
        self.check()
        now = self._clock()
        # Every chunk checks cancellation. Only intermediate UI publications
        # are throttled; each file still gets immediate start/end witnesses.
        if self._sid and (completed == 0 or finished or self._last is None or now - self._last >= 0.25):
            fields = {'phase': 'loading', 'message': f"Verifying {name}: {completed}/{total} bytes",
                      'elapsed_seconds': max(0.0, now - self._started), 'component': 'transformer'}
            if self._node is not None:
                self._node.progress(-1, **fields)
            else:
                payload = {'type': 'progress', 'node': self._node_id, 'progress': -1, 'status': 'running',
                           'task_id': self._task_id, 'attempt_index': self._attempt,
                           **self._server._current_run_identity_payload(), **fields,
                           'last_heartbeat_at': time.time()}
                self._server.queue_message(self._server.record_node_progress(payload))
            self._last = now
        self.check()


def artifact_verification_for_task(server, graph, node_id, sid, *, node=None, current_node=False):
    """Bind only the exact active graph; inspections never borrow another task."""
    from modiff.native_component_lora import _GRAPH_SCOPE
    scope = _GRAPH_SCOPE.get()
    task = getattr(server, 'current_task', None)
    args = task.get('args') if isinstance(task, dict) else None
    nodes = graph.get('nodes') if isinstance(graph, dict) else None
    source_node = nodes.get(node_id) if isinstance(nodes, dict) else None
    if (scope is None or not isinstance(graph, dict) or scope[0] is not graph or not isinstance(task, dict)
            or not isinstance(args, (list, tuple)) or not args or args[0] is not graph
            or not isinstance(task.get('task_id'), str) or not task['task_id']
            or (sid is not None and not isinstance(sid, str))
            or graph.get('sid') != sid or task.get('sid') != sid
            or not isinstance(source_node, dict)
            or (source_node.get('module'), source_node.get('action')) != ('modules.ModularDiffusers', 'ModelsLoader')
            or (current_node and task.get('current_node') != node_id)
            or (node is not None and (getattr(node, '_sid', None) != sid
                or getattr(server, 'node_cache', {}).get(node_id) is not node))):
        return None
    return _TaskArtifactVerification(server, task, graph, node_id, sid, node=node, current_node=current_node)


def artifact_verification_for_current_node(server, node_id, sid):
    from modiff.native_component_lora import _GRAPH_SCOPE
    scope = _GRAPH_SCOPE.get()
    if scope is None:
        return None
    return artifact_verification_for_task(server, scope[0], node_id, sid, current_node=True)


def artifact_verification_for_loader(node, weight_format):
    if (type(weight_format) is not str or weight_format not in ('fp8r', 'nvfp4')
            or not getattr(node, 'node_id', None)):
        return None
    from modiff.native_component_lora import _GRAPH_SCOPE
    scope = _GRAPH_SCOPE.get()
    if scope is None:
        return None
    from modiff.NodeBase import _server
    return artifact_verification_for_task(_server(), scope[0], node.node_id, node._sid,
                                          node=node, current_node=True)
