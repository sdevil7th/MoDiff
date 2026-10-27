"""Optional, task-bound observations of the built-in adapter application.

This is evidence, not an executor or an adapter inventory API. Only the real
loader stages resolved descriptors, after official activation and scheduler
application. Ordinary dispatch seals them after output and owner completion.
"""

from copy import deepcopy
import hashlib
import math
import re
import sys
import uuid
import weakref

from modiff.block_definition_v2 import _stable_json


MAX_ADAPTERS = 32
MAX_NAME = 128
MAX_REPOSITORY = 256
MAX_WEIGHT_NAME = 256
MAX_SCHEDULER_CLASS = 128
MAX_PROJECTION_BYTES = 32_768
STATE_PREFIX = 'sha256:applied-adapters-v1:'
_HEX64 = re.compile(r'[0-9a-f]{64}\Z')
_REVISION = re.compile(r'[0-9a-f]{40}\Z')
_NAME = re.compile(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*\Z')
_CLASS_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')


def unavailable(reason='unobserved-application'):
    return {'schemaVersion': 1, 'status': 'unavailable', 'reason': reason}


def _text(value, limit):
    return (type(value) is str and 0 < len(value) <= limit and value == value.strip()
            and not any(ord(c) < 32 for c in value))


def _hash(value):
    return type(value) is str and _HEX64.fullmatch(value) is not None


def project_applied_adapters(resolved):
    """Project genuine resolved descriptors without paths or runtime objects.

    Historical descriptor digests are opaque. This additional state digest uses
    the existing cross-language canonical JSON contract, including 0.0 == 0.
    Unsupported metadata makes evidence unavailable; it cannot change loading.
    """
    from modiff.auxiliary_lora import (
        ResolvedLoraDescriptor, _exact_repository, _exact_weight_name, _validate_json_shape,
    )

    if type(resolved) not in (list, tuple) or len(resolved) > MAX_ADAPTERS:
        return None
    adapters = []
    names = set()
    for item in resolved:
        if (type(item) is not ResolvedLoraDescriptor
                or not _text(item.adapter_name, MAX_NAME) or _NAME.fullmatch(item.adapter_name) is None
                or item.adapter_name in names
                or type(item.scale) not in (int, float) or not -20 <= item.scale <= 20
                or not math.isfinite(item.scale) or not _hash(item.descriptor_sha256)
                or not _hash(item.content_sha256) or not _text(item.artifact_weight_name, MAX_WEIGHT_NAME)):
            return None
        names.add(item.adapter_name)
        try:
            _exact_weight_name(item.artifact_weight_name)
            if item.source == 'hub':
                _exact_repository(item.repository)
        except (TypeError, ValueError):
            return None
        artifact = {'source': item.source, 'weightName': item.artifact_weight_name, 'sha256': item.content_sha256}
        if item.source == 'hub':
            if (not _text(item.repository, MAX_REPOSITORY) or type(item.revision) is not str
                    or _REVISION.fullmatch(item.revision) is None):
                return None
            artifact.update(repository=item.repository, revision=item.revision)
        elif item.source != 'local':
            return None
        scheduler = None
        if item.scheduler_class_name is None and (type(item.scheduler_config) is not dict or item.scheduler_config):
            return None
        if item.scheduler_class_name is not None:
            if (not _text(item.scheduler_class_name, MAX_SCHEDULER_CLASS)
                    or _CLASS_NAME.fullmatch(item.scheduler_class_name) is None or type(item.scheduler_config) is not dict):
                return None
            try:
                _validate_json_shape(item.scheduler_config, description='Applied adapter scheduler')
                config = _stable_json(item.scheduler_config).encode('utf-8')
            except (TypeError, ValueError, UnicodeError, OverflowError):
                return None
            if len(config) > 16_384:
                return None
            scheduler = {'className': item.scheduler_class_name,
                         'configSha256': 'sha256:' + hashlib.sha256(config).hexdigest()}
        scale = int(item.scale) if item.scale == int(item.scale) else item.scale
        adapters.append({'adapterName': item.adapter_name, 'scale': scale,
                         'descriptorSha256': item.descriptor_sha256, 'artifact': artifact, 'scheduler': scheduler})
    try:
        canonical = _stable_json({'adapters': adapters}).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, OverflowError):
        return None
    if len(canonical) > MAX_PROJECTION_BYTES:
        return None
    return {'adapters': adapters, 'stateSha256': STATE_PREFIX + hashlib.sha256(canonical).hexdigest()}


def _supported_owner(node):
    module = sys.modules.get('modules.ModularDiffusers.loaders')
    return (module is not None and type(node) is vars(module).get('ModelsLoader')
            and getattr(node, '_applied_adapter_builtin', False) is True)


def _publish(node, value):
    record = getattr(node, '_execution_input_record', None)
    if type(record) is dict:
        record['appliedAdapters'] = deepcopy(value)


def _same_owner(node, state, manager):
    from modiff.native_disk_lora import disk_lora_binding_matches
    from modiff.node_cache_identity import implementation_identity, input_snapshot

    if (type(state) is not dict or state['owner'] != node.node_id
            or state['pipeline']() is not getattr(node, 'loader', None)
            or state['outputIdentity'] != id(getattr(node, 'output', None))
            or state['outputSnapshot'] != input_snapshot(getattr(node, 'output', None))
            or state['implementation'] != implementation_identity(node)
            or not getattr(node, '_cache_valid', False) or getattr(node, '_cache_invalidated', False)):
        return False
    # Reuse has stronger requirements than observing one successful activation.
    # Only the existing exact native/selected-component owner seal can attest
    # unchanged resident adapters. Never invoke a supplier to reconstruct it.
    policy = state['policy']
    try:
        return policy is not None and disk_lora_binding_matches(node, policy, manager)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def begin_adapter_observation(node, manager):
    """Remove the previous task's projection before even preflight can fail."""
    state = getattr(node, '_applied_adapter_state', None)
    node._applied_adapter_reuse_valid = _supported_owner(node) and _same_owner(node, state, manager)
    node._applied_adapter_pending = None
    node._applied_adapter_call_complete = False
    _publish(node, unavailable())


def abort_adapter_observation(node):
    node._applied_adapter_pending = None
    node._applied_adapter_state = None
    node._applied_adapter_call_complete = False
    _publish(node, unavailable('owner-invalidated'))


def stage_adapter_observation(node, resolved):
    """Called only after actual activation/removal, scheduler and loader success."""
    projection = project_applied_adapters(resolved)
    try:
        pipeline = weakref.ref(node.loader)
    except TypeError:
        projection, pipeline = None, None
    node._applied_adapter_pending = projection, pipeline


def seal_adapter_observation(node, manager):
    """Join the observed application to this successful ordinary dispatch."""
    if not _supported_owner(node):
        node._applied_adapter_state = None
        _publish(node, unavailable('unsupported-owner'))
        return
    if not getattr(node, '_applied_adapter_call_complete', False):
        abort_adapter_observation(node)
        return
    pending = getattr(node, '_applied_adapter_pending', None)
    node._applied_adapter_pending = None
    if pending is not None:
        projection, pipeline = pending
        if projection is None:
            node._applied_adapter_state = None
            _publish(node, unavailable('unsupported-projection'))
            return
        if (pipeline() is not getattr(node, 'loader', None) or not getattr(node, '_cache_valid', False)
                or getattr(node, '_cache_invalidated', False) or type(getattr(node, 'output', None)) is not dict):
            abort_adapter_observation(node)
            return
        from modiff.node_cache_identity import implementation_identity, input_snapshot
        native = getattr(node, '_native_disk_lora_state', None)
        state = {'owner': node.node_id, 'pipeline': pipeline, 'outputIdentity': id(node.output),
                 'outputSnapshot': input_snapshot(node.output),
                 'implementation': implementation_identity(node),
                 'policy': native.get('policy') if type(native) is dict else None,
                 'projection': projection, 'applicationId': uuid.uuid4().hex}
        # Any selected/native lifecycle must have completed its existing seal.
        if state['policy'] is not None and not _same_owner(node, state, manager):
            abort_adapter_observation(node)
            return
        node._applied_adapter_state = state
        observation = 'activated'
    else:
        state = getattr(node, '_applied_adapter_state', None)
        if (getattr(node, '_cache_reason', None) not in {'unchanged_inputs', 'usage_changed'}
                or not getattr(node, '_applied_adapter_reuse_valid', False) or not _same_owner(node, state, manager)):
            reason = ('unobserved-application' if state is None else
                      'unsupported-owner' if state['policy'] is None else 'owner-invalidated')
            node._applied_adapter_state = None
            _publish(node, unavailable(reason))
            return
        observation = 'reused'
    _publish(node, {'schemaVersion': 1, 'status': 'complete', 'source': 'backend-adapter-application',
                    'applicationId': state['applicationId'], 'observation': observation, **state['projection']})
