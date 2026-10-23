"""Serialized, server-observed native ownership and exact terminal release.

Purpose and Block context select an association; only a successful dispatcher
can seal a weak owner incarnation. Neither client hint grants eviction power.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
import sys
import threading
import weakref

_SCOPE = ContextVar('modiff_ancillary_owner_scope', default=None)


class AncillaryOwnerError(ValueError):
    modiff_error_code = 'ancillary_owner_boundary'
    modiff_category = 'execution'
    modiff_recovery_hint = 'Preserved model owners cannot be replaced by this Preview. Use normal explicit idle cleanup, then Run again.'


def _error(message):
    return AncillaryOwnerError('Ancillary owner boundary: ' + message)


def _worker(server):
    return (id(server), os.getpid(), server.instance,
            server.backend_source_identity.get('fingerprint'))


def _leased(server, graph):
    task = getattr(server, 'current_task', None)
    args = task.get('args') if type(task) is dict else None
    binding = getattr(server, '_model_executor_thread_binding', None)
    lock = getattr(server, '_node_cache_lock', None)
    return bool(type(task) is dict and type(task.get('task_id')) is str and task['task_id']
        and isinstance(args, (tuple, list)) and args and args[0] is graph
        and task.get('sid') == graph.get('sid')
        and lock is not None and lock.locked() and isinstance(binding, tuple)
        and binding == (id(server._model_executor), threading.get_ident())
        and (not server.serialize_model_io or server.model_io_lock.locked())
        and type(server.instance) is str and server.instance
        and type(server.backend_source_identity.get('fingerprint')) is str
        and server.backend_source_identity['fingerprint'])


def observe_claimed_attempt(server, graph):
    """Preservation only, before a queued claim can fail dispatch validation.

    This marker is minted from an actual leased callback, not deserialized from
    hints. It prevents the worker's historical global failure cleanup. It grants
    no release, allocation or retry authority.
    """
    task = getattr(server, 'current_task', None)
    queued = getattr(server, 'task_graphs', {}).get(task.get('task_id')) if type(task) is dict else None
    queued_hints = queued.get('runtimeHints') if type(queued) is dict else None
    queued_claim = queued_hints.get('executionPurpose') if type(queued_hints) is dict else None
    issued_preview = type(queued_claim) is dict and queued_claim.get('kind') == 'explicit_preview'
    hints = graph.get('runtimeHints') if type(graph) is dict else None
    claim = hints.get('executionPurpose') if type(hints) is dict else None
    if not issued_preview and (type(hints) is not dict or 'executionPurpose' not in hints
            or type(claim) is dict and claim.get('kind') == 'workflow'):
        return
    if _leased(server, graph):
        task = server.current_task
        server._ancillary_owner_attempt = (id(task), task['task_id'], id(graph), _worker(server))
        if issued_preview:
            from modiff.workflow_auto_resource import workflow_graph_hash
            try:
                unchanged = claim == queued_claim and workflow_graph_hash(graph) == workflow_graph_hash(queued)
            except (ValueError, TypeError):
                unchanged = False
            if not unchanged:
                raise _error('the server-queued Preview selection or graph changed before dispatch.')


def preserve_failed_attempt(server):
    task = getattr(server, 'current_task', None)
    args = task.get('args') if type(task) is dict else None
    marker = getattr(server, '_ancillary_owner_attempt', None)
    # Drift prohibits release, but cannot authorize global destruction. Preserve
    # the exact issued task even when its instance/source observation changed.
    return bool(type(task) is dict and isinstance(args, (tuple, list)) and args
        and type(marker) is tuple and len(marker) == 4
        and marker[:2] == (id(task), task.get('task_id'))
        and marker[3][:2] == (id(server), os.getpid()))


def clear_attempt(server):
    server.__dict__.pop('_ancillary_owner_attempt', None)


def _digest(value):
    # NodeBase snapshots contain primitive identities, types and code objects,
    # never a retained tensor/model. Do not copy model inputs or weights.
    return hashlib.sha256(repr(value).encode('utf-8')).hexdigest()


def _parent_identity(value):
    # JSON object member order is not part of the selected parent contract.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class _DataOrigin:
    node: object
    output_id: int
    output_digest: str
    inputs: str
    implementation: str
    sources: frozenset


@dataclass(frozen=True)
class _Origin:
    node: object
    output_id: int
    inputs: str
    implementation: str
    sources: frozenset
    manager: tuple
    components: tuple
    worker: tuple
    task_id: str
    attempt: int
    parent: str
    graph_hash: str
    association: str | None = None
    supporting: tuple = ()


def _observe(server, node_id, parent, graph_hash, association=None):
    from utils.memory_menager import memory_manager
    node = server.node_cache.get(node_id)
    if (node is None or getattr(node, '_cache_valid', False) is not True
            or getattr(node, '_cache_invalidated', False)):
        raise _error('only a successful current dispatched owner can be observed.')
    identity = (getattr(node, 'module_name', None), getattr(node, 'class_name', None))
    loaded = sys.modules.get(identity[0] + '.main') if type(identity[0]) is str else None
    action = vars(loaded).get(identity[1]) if loaded is not None else None
    if not isinstance(action, type) or type(node) is not action:
        raise _error('only the exact preloaded native node can issue an owner observation.')
    manager = []
    for model_id in getattr(node, '_mm_models', ()):
        model = memory_manager.cache.get(model_id, {}).get('model')
        if model is None:
            raise _error('a dispatched manager owner is missing.')
        manager.append((model_id, weakref.ref(model)))
    native = getattr(sys.modules.get('modules.ModularDiffusers'), 'components', None)
    entries = getattr(native, 'components', {})
    components = tuple((key, weakref.ref(entries[key])) for key in
        getattr(native, 'collections', {}).get(node_id, ()) if key in entries)
    supporting, seen = [], set()
    pending = list(getattr(node, '_cache_input_sources', ()))
    from modiff.workflow_auto_lifecycle import _material
    while pending:
        source_id = pending.pop()
        if source_id in seen:
            continue
        seen.add(source_id)
        source = server.node_cache.get(source_id)
        loaded = sys.modules.get('modules.ModularDiffusers.main')
        action = vars(loaded).get('Lora') if loaded is not None else None
        if (type(source) is not action or not getattr(source, '_cache_valid', False)
                or getattr(source, '_cache_invalidated', False) or getattr(source, '_mm_models', ())
                or source_id in getattr(native, 'collections', {})):
            continue  # Unknown/custom suppliers never gain release authority.
        sources = frozenset(getattr(source, '_cache_input_sources', ()))
        supporting.append((source_id, _DataOrigin(weakref.ref(source), id(source.output),
            _digest(_material(source.output)), _digest(getattr(source, '_cache_input_snapshot', None)),
            _digest(getattr(source, '_cache_implementation', None)), sources)))
        pending.extend(sources)
    task = server.current_task
    return _Origin(weakref.ref(node), id(node.output), _digest(getattr(node, '_cache_input_snapshot', None)),
        _digest(getattr(node, '_cache_implementation', None)),
        frozenset(getattr(node, '_cache_input_sources', ())), tuple(manager), components,
        _worker(server), task['task_id'], int(task.get('attempt_index') or 0), parent, graph_hash,
        association, tuple(supporting))


def _no_modular_owners():
    manager = getattr(sys.modules.get('modules.ModularDiffusers'), 'components', None)
    return not any(getattr(manager, name, None) for name in ('components', 'collections', 'model_hooks'))


class _Scope:
    def __init__(self, server, graph, selection):
        from copy import deepcopy
        from modiff.workflow_auto_resource import workflow_graph_hash
        self.server, self.graph = server, graph
        self.selection = deepcopy(selection)
        self.preview = selection['kind'] == 'explicit_preview'
        self.worker, self.task = _worker(server), id(server.current_task)
        self.attempt = int(server.current_task.get('attempt_index') or 0)
        self.graph_hash = workflow_graph_hash(graph)
        self.origins = {}
        self.owner_bindings = {}
        self.pending_turnover = None
        self.preallocation_witness = None
        self.auto_owner = None
        self.auto = False
        self.initial_cache_ids = frozenset(server.node_cache)
        self.finished = False
        if self.preview:
            hints = graph['runtimeHints']
            self.auto = (hints.get('resourceMode') == 'auto'
                and type(hints.get('workflowAutoPlan')) is dict
                and not any(name in hints for name in ('autoResourcePlan', 'resourcePlan')))
            explicit = hints.get('resourceMode') == 'expert' and not any(name in hints for name in
                ('workflowAutoPlan', 'autoResourcePlan', 'resourcePlan'))
            if not explicit and not self.auto:
                raise _error('Preview requires explicit Custom memory; no reviewed Auto or incoming budget exists.')
            self.binding = selection['bindings'][0]
            self.provider = self.binding['roleNodes']['provider']
            self.proposal = self.binding['roleNodes']['proposal']
            self.pending_turnover = self._turnover_event()
            if self.pending_turnover is None:
                self._empty_start()

    def bind_auto_plan(self, plan):
        """Accept only the plan just recomputed by the actual server dispatcher.

        The request receipt selects a normal planning pass; it never supplies a
        budget or a release grant. The concrete recipe remains source-closed
        until its complete resource envelope has been reviewed.
        """
        from copy import deepcopy
        from modiff.transformers_image_text_resources import image_text_cpu_preparation_requirements
        self.check()
        owners = plan.get('loaders', [])
        def budget(value, fields):
            return (type(value) is dict and set(value) == set(fields)
                and all(type(amount) is int and 0 <= amount < 2**63 for amount in value.values()))
        owner = owners[0] if type(owners) is list and len(owners) == 1 else {}
        provider = self.graph['nodes'][self.provider]['params']
        settings = owner.get('settings', {})
        complete = (budget(owner.get('requirements'), ('systemRamBytes', 'vramBytes', 'diskFreeBytes'))
            and budget(owner.get('parameterResidencyRequirements'), ('systemRamBytes', 'vramBytes'))
            and budget(owner.get('parameterLoadingRequirements'), ('systemRamBytes', 'vramBytes'))
            and owner.get('capacityRequirements') == {}
            and owner.get('consumers') == [self.proposal]
            and owner.get('workingMemoryPolicy') == 'runtime_headroom_policy'
            and owner.get('workingDemandEvidence', {}).get('cpuPreparationRequirements') ==
                image_text_cpu_preparation_requirements()
            and settings.get('device') == provider.get('device', {}).get('value')
            and settings.get('dtype') == provider.get('dtype', {}).get('value')
            and settings.get('quantizationMode') == provider.get('quantization_mode', {}).get('value'))
        if (not self.auto or not plan.get('canAutoRun') or plan.get('patches')
                or plan.get('schedule') or plan.get('requiresPreparation')
                or plan.get('graphHash') != self.graph_hash or plan.get('plannedGraphHash') != self.graph_hash
                or not complete or owner.get('nodeId') != self.provider
                or owner.get('modelType') != 'ImageTextToText'
                or owner.get('workingDemandEvidence', {}).get('kind') != 'reviewed_bounded_image_text_storage'):
            raise _error('Preview has no exact complete server-computed Auto owner budget.')
        self.auto_owner = deepcopy(owners[0])

    def defer_auto_cache(self, plan):
        self.bind_auto_plan(plan)
        # A projected reclamation is not free capacity. Destruction waits for
        # the exact processor/input witness and scoped ownership event below.
        return plan, {'performed': False, 'scopedUntilProcessorPreflight': True}

    def check(self):
        from modiff.workflow_auto_resource import workflow_graph_hash
        server = self.server
        if (not _leased(server, self.graph) or self.worker != _worker(server)
                or id(server.current_task) != self.task
                or int(server.current_task.get('attempt_index') or 0) != self.attempt
                or self.selection != self.graph.get('runtimeHints', {}).get('executionPurpose')
                or self.graph_hash != workflow_graph_hash(self.graph)):
            raise _error('the actual worker, task, graph or selected parent changed.')

    def _empty_start(self):
        from types import SimpleNamespace
        from modiff.workflow_auto_lifecycle import _RetainedPromptProposal
        from utils.memory_menager import memory_manager
        if memory_manager.cache or any(memory_manager._active_ids.values()) or not _no_modular_owners():
            raise _error('resident or foreign model ownership has no native Block turnover/admission grant.')
        # The first slice supports only a cold owner. Exact selected Load Image
        # sources and our detached previous proposal can remain cached. No class
        # property, arbitrary supplier or model is invoked during this check.
        for node_id, node in self.server.node_cache.items():
            if self._incoming_node(node_id, node):
                continue
            retained = getattr(self.server, '_ancillary_detached_material', {}).get(node_id)
            if (type(node) in (SimpleNamespace, _RetainedPromptProposal) and retained == (id(node), id(node.output))
                    and not vars(node).get('_mm_models') and vars(node).get('params') == {}):
                continue
            expected = self.graph['nodes'].get(node_id)
            loaded = sys.modules.get('modules.Image.main')
            media_action = vars(loaded).get('Load') if loaded is not None else None
            media_candidate = (type(expected) is dict
                and (expected['module'], expected['action']) == ('modules.Image', 'Load')
                and type(node) is media_action)
            if type(node) not in (SimpleNamespace, _RetainedPromptProposal) and not media_candidate:
                raise _error('unknown or foreign cached owner has no private origin association.')
            state = vars(node)
            key = (state.get('module_name'), state.get('class_name'))
            same = type(expected) is dict and key == (expected['module'], expected['action'])
            loaded = sys.modules.get(key[0] + '.main') if type(key[0]) is str else None
            action = vars(loaded).get(key[1]) if loaded is not None else None
            media = same and key == ('modules.Image', 'Load') and type(node) is action
            detached = (node_id == self.proposal and key == ('modules.HuggingFaceTransformers', 'GenerateImageVideoText')
                and type(node) in (SimpleNamespace, _RetainedPromptProposal) and type(state.get('output')) is dict
                and type(state['output'].get('proposal_summary')) is str and state.get('params') == {})
            if state.get('_mm_models') or not (media or detached):
                raise _error('unknown or foreign cached owner has no private origin association; no image eviction is permitted.')

    def _incoming_node(self, node_id, node):
        loaded = sys.modules.get('modules.HuggingFaceTransformers.main')
        action = vars(loaded).get('LoadImageTextToTextModel') if loaded is not None else None
        return (node_id == self.provider and type(node) is action
            and self.server.current_task.get('current_node') == self.provider
            and not getattr(node, '_mm_models', ()) and not getattr(node, '_cache_valid', False))

    def validate_owner(self, node_id, *, origin=None, historical=False):
        from modiff.node_cache_identity import implementation_identity
        from utils.memory_menager import memory_manager
        origin = self.origins.get(node_id) if origin is None else origin
        if origin is None:
            raise _error('missing private dispatched owner origin.')
        node = self.server.node_cache.get(node_id)
        if (node is not origin.node() or node is None or id(node.output) != origin.output_id
                or getattr(node, '_cache_valid', False) is not True or getattr(node, '_cache_invalidated', False)
                or _digest(getattr(node, '_cache_input_snapshot', None)) != origin.inputs
                or _digest(getattr(node, '_cache_implementation', None)) != origin.implementation
                or _digest(implementation_identity(node)) != origin.implementation
                or frozenset(getattr(node, '_cache_input_sources', ())) != origin.sources
                or tuple(getattr(node, '_mm_models', ())) != tuple(key for key, _ref in origin.manager)
                or origin.worker != self.worker
                or (not historical and (origin.task_id != self.server.current_task['task_id']
                    or origin.parent != _parent_identity(self.binding['parent'])
                    or origin.attempt != self.attempt or origin.graph_hash != self.graph_hash))):
            raise _error('the exact dispatched owner incarnation changed.')
        for model_id, reference in origin.manager:
            if reference() is None or memory_manager.cache.get(model_id, {}).get('model') is not reference():
                raise _error('the exact manager record changed.')
            if memory_manager._active_ids.get(model_id, 0):
                raise _error('a selected model is still active.')
        if origin.components and not historical:
            raise _error('this slice cannot release an image/native component owner.')
        if historical:
            native = getattr(sys.modules.get('modules.ModularDiffusers'), 'components', None)
            entries = getattr(native, 'components', {})
            if set(getattr(native, 'collections', {}).get(node_id, ())) != {key for key, _ in origin.components}:
                raise _error('the native component collection changed.')
            if any(reference() is None or entries.get(key) is not reference() for key, reference in origin.components):
                raise _error('the native component incarnation changed.')
            from modiff.workflow_auto_lifecycle import _material
            for source_id, record in origin.supporting:
                source = self.server.node_cache.get(source_id)
                if (source is None or source is not record.node() or id(source.output) != record.output_id
                        or not getattr(source, '_cache_valid', False) or getattr(source, '_cache_invalidated', False)
                        or getattr(source, '_mm_models', ()) or source_id in getattr(native, 'collections', {})
                        or _digest(_material(source.output)) != record.output_digest
                        or _digest(getattr(source, '_cache_input_snapshot', None)) != record.inputs
                        or _digest(getattr(source, '_cache_implementation', None)) != record.implementation
                        or _digest(implementation_identity(source)) != record.implementation
                        or frozenset(getattr(source, '_cache_input_sources', ())) != record.sources):
                    raise _error('the exact dispatched LoRA descriptor supplier changed.')

    def _turnover_event(self):
        """Validate in its own frame so no strong neural references cross release."""
        from modiff.ancillary_execution import native_parent_association
        from utils.memory_menager import memory_manager
        native = getattr(sys.modules.get('modules.ModularDiffusers'), 'components', None)
        resident = bool(memory_manager.cache or not _no_modular_owners())
        if not resident:
            return None
        association = native_parent_association(self.graph, self.binding, self.server.modules)
        owner_id = self.binding['parentRoleNodes']['models']
        origin = getattr(self.server, '_ancillary_dispatched_origins', {}).get(owner_id)
        if (association is None or type(origin) is not _Origin or origin.association != association
                or origin.parent != _parent_identity(self.binding['parent'])):
            raise _error('resident ownership has no completed matching native Block origin association.')
        self.validate_owner(owner_id, origin=origin, historical=True)
        supporting = {key for key, _record in origin.supporting}
        targets = set(self.server._dependent_cache_nodes([owner_id])) | supporting
        selected = set(self.binding['parentRoleNodes'].values()) | supporting
        if not targets <= selected or set(memory_manager.cache) != {key for key, _ in origin.manager}:
            raise _error('foreign or retaining owners prevent native Block turnover.')
        collections = getattr(native, 'collections', {})
        expected = {key for key, _ in origin.components}
        if (set(getattr(native, 'components', {})) != expected
                or any(set(keys) & expected for key, keys in collections.items() if key not in targets)
                or set(collections) - targets or any(memory_manager._active_ids.values())):
            raise _error('shared, active or foreign native component ownership prevents turnover.')
        for node_id in set(self.server.node_cache) - targets:
            node = self.server.node_cache[node_id]
            if self._incoming_node(node_id, node):
                continue  # The current incoming dispatcher frame owns no model.
            retained = getattr(self.server, '_ancillary_detached_material', {}).get(node_id)
            definition = self.graph['nodes'].get(node_id, {})
            loaded = sys.modules.get('modules.Image.main')
            action = vars(loaded).get('Load') if loaded is not None else None
            selected_media = (node_id in selected and type(node) is action and
                (definition.get('module'), definition.get('action')) == ('modules.Image', 'Load'))
            if not selected_media and retained != (id(node), id(getattr(node, 'output', None))):
                raise _error('foreign cached state prevents native Block turnover.')
        return {'ownerIds': [owner_id], 'nodeIds': sorted(targets),
                'retainOutputs': {}, 'retainMaterialOutputs': True}

    def before_node(self, node_id):
        self.check()
        if not self.preview:
            return
        if self.auto and node_id == self.provider:
            if self.auto_owner is None:
                raise _error('Preview has no server-computed processor preparation RAM reservation.')
            from modiff.workflow_auto_lifecycle import assert_next_owner_capacity
            from modiff.transformers_image_text_resources import image_text_cpu_preparation_requirements
            preparation = image_text_cpu_preparation_requirements()
            try:
                assert_next_owner_capacity(self.server, {
                    'ownerId': self.provider, 'requirements': preparation,
                    'capacityRequirements': {}, 'workingMemoryPolicy': 'runtime_headroom_policy',
                    'parameterResidencyRequirements': {'systemRamBytes': 0, 'vramBytes': 0},
                    'parameterLoadingRequirements': {'systemRamBytes': preparation['systemRamBytes'], 'vramBytes': 0},
                    'device': 'cpu', 'offloadMode': 'none'})
            except ValueError as error:
                raise _error('actual RAM before processor preparation is below its complete CPU reservation; the outgoing owner was preserved.') from error
            self.check()
        if self.pending_turnover is not None:
            self._turnover_event()
            return  # Explicit processor preflight must pass before destruction.

        from utils.memory_menager import memory_manager
        allowed = set()
        for owner_id, origin in self.origins.items():
            self.validate_owner(owner_id)
            allowed.update(key for key, _reference in origin.manager)
        if (set(memory_manager.cache) != allowed or any(memory_manager._active_ids.values())
                or set(self.server.node_cache) - self.initial_cache_ids - set(self.graph['nodes'])
                or not _no_modular_owners()):
            raise _error('foreign ownership appeared before allocation; historical pressure eviction is not permitted.')
        if node_id not in self.graph['nodes']:
            raise _error('dispatch left its exact selected graph.')

    def before_factory(self, processor, selection, revision):
        self.check()
        if self.auto and self.auto_owner is None:
            raise _error('Preview reached its factory without its server-computed Auto budget.')
        if self.server.current_task.get('current_node') != self.provider:
            raise _error('processor preflight is outside the exact incoming provider dispatch.')
        from modiff.ancillary_execution import adapter_binding
        descriptor, _declaration = adapter_binding()
        if selection != {'source': 'hub', 'value': descriptor['repository']} or revision != descriptor['revision']:
            raise _error('processor preflight differs from the immutable selected provider.')
        request = self.graph['nodes'][self.proposal]
        values = {key: param.get('value') for key, param in request['params'].items()
                  if param.get('sourceId') is None}
        image = request['params'].get('images', {})
        if image.get('sourceId'):
            source = self.server.node_cache.get(image['sourceId'])
            loaded = sys.modules.get('modules.Image.main')
            action = vars(loaded).get('Load') if loaded is not None else None
            if type(source) is not action or not getattr(source, '_cache_valid', False):
                raise _error('the real selected reference must be dispatched before provider allocation.')
            values['images'] = source.output[image['sourceKey']]
        from modiff.prompt_enhancement import preflight_processor_inputs
        fields, limits = preflight_processor_inputs(processor, values, self)
        self.preallocation_witness = (weakref.ref(processor), fields, limits, self.graph_hash)
        if self.pending_turnover is not None:
            self._turnover_event()
            from modiff.workflow_auto_lifecycle import release_owner_caches
            from utils.memory_menager import memory_manager
            receipt = release_owner_caches(self.server, self.pending_turnover, memory_manager)
            self.pending_turnover = None
            self.server.__dict__.pop('_ancillary_dispatched_origins', None)
            detached = getattr(self.server, '_ancillary_detached_material', {})
            detached.update({key: (id(self.server.node_cache[key]), id(self.server.node_cache[key].output))
                for key in receipt['retainedOutputNodes']})
            self.server._ancillary_detached_material = detached
            self._empty_start()
        if self.auto:
            # Recompute against actual post-release memory and the unchanged
            # exact artifact/request. No cache projection is fit evidence and
            # no global cleanup or Expert retry is permitted at this boundary.
            refreshed = self.server._build_workflow_auto_plan(self.graph, dispatch=True)
            if refreshed.get('requiresCachePreparation'):
                raise _error('actual memory after scoped release still requires cache preparation.')
            previous = self.auto_owner
            self.bind_auto_plan(refreshed)
            if self.auto_owner != previous:
                raise _error('the exact incoming Auto budget changed after processor preflight.')
            from modiff.workflow_auto_lifecycle import assert_next_owner_capacity
            owner = self.auto_owner
            assert_next_owner_capacity(self.server, {
                'ownerId': self.provider, 'requirements': owner['requirements'],
                'capacityRequirements': owner['capacityRequirements'],
                'workingMemoryPolicy': owner['workingMemoryPolicy'],
                'parameterResidencyRequirements': owner['parameterResidencyRequirements'],
                'parameterLoadingRequirements': owner['parameterLoadingRequirements'],
                'device': owner['settings']['device'], 'offloadMode': owner['settings']['offloadMode']})
            self.check()

    def check_inputs(self, node, processor, fields, limits):
        self.check()
        witness = self.preallocation_witness
        if (node.node_id != self.proposal or self.server.current_task.get('current_node') != self.proposal
                or witness is None or witness[0]() is not processor or witness[1] != fields
                or witness[2] != limits or witness[3] != self.graph_hash):
            raise _error('the actual processed request changed after per-run preallocation.')

    def dispatched(self, node_id):
        self.check()
        if self.preview:
            if node_id != self.provider:
                return
            parent = _parent_identity(self.binding['parent'])
        else:
            selected = [binding for binding in self.selection['bindings']
                        if binding['parentRoleNodes'].get('models') == node_id]
            if len(selected) != 1:
                return
            parent = _parent_identity(selected[0]['parent'])
        from modiff.ancillary_execution import native_parent_association
        association = None if self.preview else native_parent_association(
            self.graph, selected[0], self.server.modules)
        origin = _observe(self.server, node_id, parent, self.graph_hash, association)
        self.origins[node_id] = origin
        if not self.preview:
            self.owner_bindings[node_id] = selected[0]
        # Observations are weak, bounded by the current graph and grant no image
        # eviction. A new task replaces this private diagnostic inventory.
        if self.preview:
            self.server._ancillary_dispatched_origins = dict(self.origins)

    def finish(self):
        if not self.preview:
            self.check()
            # Publication occurs only after every real stage returned, not
            # merely after the loader succeeded. Keep weak records only.
            if not self.server.interrupt_flag and not self.server.current_task.get('interrupt_requested'):
                from modiff.ancillary_execution import native_parent_association
                for node_id, origin in self.origins.items():
                    self.validate_owner(node_id, origin=origin, historical=True)
                    if origin.association != native_parent_association(
                            self.graph, self.owner_bindings[node_id], self.server.modules):
                        raise _error('the native Block association changed before completion.')
                self.server._ancillary_dispatched_origins = dict(self.origins)
            return
        self.check()
        if self.finished or self.server.interrupt_flag or self.server.current_task.get('interrupt_requested'):
            raise _error('interrupted or already consumed terminal owner event.')
        self.validate_owner(self.provider)
        targets = set(self.server._dependent_cache_nodes([self.provider]))
        if (targets != {self.provider, self.proposal}
                or set(self.server.node_cache) - self.initial_cache_ids - set(self.graph['nodes'])):
            raise _error('a foreign retaining dependent prevents terminal owner release.')
        proposal = self.server.node_cache.get(self.proposal)
        summary = getattr(proposal, 'output', {}).get('proposal_summary')
        if type(summary) is not str or not summary or len(summary.encode('utf-8')) > 2 * 1024**2:
            raise _error('terminal self-release requires the bounded native proposal summary.')
        proposal = None
        origin = self.origins[self.provider]
        ids = {key for key, _reference in origin.manager}
        from utils.memory_menager import memory_manager
        if set(memory_manager.cache) != ids or not _no_modular_owners() or any(
                ids.intersection(getattr(node, '_mm_models', ()))
                for node_id, node in self.server.node_cache.items() if node_id not in targets):
            raise _error('a shared or foreign model owner prevents terminal release.')
        self.finished = True
        self.server.__dict__.pop('_ancillary_dispatched_origins', None)
        from modiff.workflow_auto_lifecycle import release_owner_caches
        from modiff.prompt_enhancement_summary import terminal_cache_observation, bind_terminal_detached_observation
        observed = terminal_cache_observation(self.server, self.proposal)
        retained_outputs = ['proposal_summary', 'text', 'result']
        if observed is not None and observed.pending.diagnostic_sha256 is not None:
            from modiff.prompt_enhancement_summary import digest_bytes, require_native_cache_output
            diagnostic = require_native_cache_output(self.server, self.proposal, field='proposal_diagnostic')
            if digest_bytes(diagnostic.encode('utf-8')) != observed.pending.diagnostic_sha256:
                raise _error('the observed terminal proposal diagnostic changed before release.')
            retained_outputs.append('proposal_diagnostic')
        receipt = release_owner_caches(self.server, {'ownerIds': [self.provider], 'nodeIds': sorted(targets),
            'retainOutputs': {self.proposal: retained_outputs}}, memory_manager)
        if observed is not None:
            bind_terminal_detached_observation(self.server, self.proposal, receipt, observed)
        return receipt


@contextmanager
def owner_scope(server, graph, selection):
    if selection is None:
        yield
        return
    if not _leased(server, graph):
        if selection['kind'] == 'explicit_preview':
            raise _error('a Preview owner requires the actual serialized graph callback.')
        yield  # Existing direct/manual workflow tests issue no observations.
        return
    scope = _Scope(server, graph, selection)
    token = _SCOPE.set(scope)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def active_preview(server):
    scope = _SCOPE.get()
    return scope is not None and scope.server is server and scope.preview


def bind_preview_auto_plan(server, plan):
    scope = _SCOPE.get()
    if scope is not None and scope.server is server and scope.preview:
        scope.bind_auto_plan(plan)


def defer_preview_auto_cache(server, plan):
    scope = _SCOPE.get()
    if scope is not None and scope.server is server and scope.preview:
        return scope.defer_auto_cache(plan)
    return None


def preview_execution_paths(server, paths):
    """Order the real selected file supplier before explicit processor preflight."""
    scope = _SCOPE.get()
    if scope is None or scope.server is not server or not scope.preview:
        return paths
    source = scope.binding['parentRoleNodes'].get('loadImage')
    if source is None:
        return paths
    order = list(dict.fromkeys(node_id for path in paths for node_id in path))
    if source in order and scope.provider in order:
        order.remove(source)
        order.insert(order.index(scope.provider), source)
    return [order]


def before_image_text_factory(processor, *, selection, revision):
    scope = _SCOPE.get()
    if scope is not None and scope.preview:
        return scope.before_factory(processor, selection, revision)


def check_preallocation_inputs(node, processor, fields, limits):
    scope = _SCOPE.get()
    if scope is not None and scope.preview:
        return scope.check_inputs(node, processor, fields, limits)


def before_node(server, node_id):
    scope = _SCOPE.get()
    if scope is not None and scope.server is server:
        scope.before_node(node_id)


def record_dispatched_owner(server, node_id):
    scope = _SCOPE.get()
    if scope is not None and scope.server is server:
        scope.dispatched(node_id)


def finish_preview(server):
    scope = _SCOPE.get()
    if scope is not None and scope.server is server:
        return scope.finish()



def image_text_auto_loader_policy():
    scope = _SCOPE.get()
    if scope is None or not scope.preview or not scope.auto:
        return {}
    scope.check()
    if scope.auto_owner is None or scope.server.current_task.get('current_node') != scope.provider:
        raise _error('native Auto loader policy is outside the exact incoming provider.')
    from modiff.transformers_image_text_resources import image_text_t2i_loader_policy
    return image_text_t2i_loader_policy()


def before_image_text_generation(model, batch, controls):
    scope = _SCOPE.get()
    if scope is None or not scope.preview or not scope.auto:
        return
    scope.check()
    if scope.server.current_task.get('current_node') != scope.proposal:
        raise _error('native Auto generation policy is outside the exact proposal.')
    from modiff.transformers_image_text_resources import validate_image_text_t2i_generation
    validate_image_text_t2i_generation(model, batch, controls)
    scope.check()
