"""Model-owner lifetimes for the existing ordered graph executor."""
from __future__ import annotations

from types import SimpleNamespace


def plan_owner_lifetimes(graph, loaders, adapters):
    # Match the executor's once-per-attempt outer DAG traversal. Shared path
    # prefixes do not extend model lifetimes or reinitialize mutable outputs.
    order = list(dict.fromkeys(node_id for path in graph['paths'] for node_id in path))
    if not graph.get('loops'):
        # Stable topological ordering: finish an already loaded owner's ready
        # consumers before opening another independent owner. The graph and
        # its data dependencies remain unchanged.
        pending, completed, reordered = set(order), set(), []
        owners = {item['nodeId']: {item['nodeId'], *item['consumers']} for item in loaders}
        dependencies = {key: {p['sourceId'] for p in node['params'].values() if p.get('sourceId')} for key, node in graph['nodes'].items()}
        while pending:
            ready = [key for key in order if key in pending and dependencies[key] <= completed]
            active_members = {key for owner, members in owners.items() if owner in completed for key in members}
            chosen = min(ready, key=lambda key: (0 if key in active_members else 2 if key in owners else 1, order.index(key)))
            reordered.append(chosen)
            completed.add(chosen)
            pending.remove(chosen)
        order = reordered
    intervals = []
    for loader in loaders:
        members = {loader['nodeId'], *loader['consumers']}
        positions = [i for i, node_id in enumerate(order) if node_id in members]
        budget = {key: int(loader['requirements'].get(key, 0)) for key in ('systemRamBytes', 'vramBytes')}
        for adapter in adapters:
            if loader['nodeId'] in adapter['loaderIds']:
                for key in budget:
                    budget[key] += adapter[key]
        intervals.append({'ownerId': loader['nodeId'], 'nodeIds': sorted(members),
                          'first': min(positions), 'last': max(positions), 'requirements': budget})
    # Loop steps execute atomically through the existing loop executor. Until a
    # loop has finished, every participating owner remains live. No release is
    # inserted into an opaque loop body.
    if graph.get('loops'):
        for interval in intervals:
            interval['first'], interval['last'] = 0, len(order) - 1
    peak = {key: 0 for key in ('systemRamBytes', 'vramBytes')}
    shared_peak = 0
    for index in range(len(order)):
        live = [item for item in intervals if item['first'] <= index <= item['last']]
        current = {key: sum(item['requirements'][key] for item in live) for key in peak}
        for key in peak:
            peak[key] = max(peak[key], current[key])
        shared_peak = max(shared_peak, sum(current.values()))
    releases = []
    for end in sorted({item['last'] for item in intervals if item['last'] < len(order) - 1}):
        expired = [item for item in intervals if item['last'] <= end]
        live_members = {node_id for item in intervals if item['last'] > end for node_id in item['nodeIds']}
        release_nodes = {node_id for item in expired for node_id in item['nodeIds']} - live_members
        already_released = {node_id for event in releases for node_id in event['nodeIds']}
        release_nodes -= already_released
        if not release_nodes:
            continue
        retained = {}
        for target in order[end + 1:]:
            for param in graph['nodes'][target]['params'].values():
                source = param.get('sourceId')
                if source in release_nodes:
                    retained.setdefault(source, set()).add(param.get('sourceKey'))
        releases.append({'afterIndex': end, 'afterNodeId': order[end],
                         'ownerIds': [item['ownerId'] for item in intervals if item['last'] == end],
                         'nodeIds': sorted(release_nodes),
                         'retainOutputs': {key: sorted(value) for key, value in retained.items()}})
    return {'executionOrder': order, 'peak': peak, 'sharedPeakBytes': shared_peak, 'owners': intervals, 'releases': releases}


def _material(value, depth=0):
    """Retain only detached data, never an opaque pipeline/model reference."""
    if depth > 32:
        raise ValueError('A retained output exceeds the data nesting limit.')
    if value is None or type(value) in {str, bool, int, float, bytes}:
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_material(item, depth + 1) for item in value)
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _material(item, depth + 1) for key, item in value.items()}
    # Imports stay lazy; these libraries are already installed by the runtime.
    from PIL.Image import Image
    import numpy as np
    if isinstance(value, Image):
        return value.copy()  # Detach custom attributes possibly holding models.
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        return np.array(value, copy=True, subok=False)
    raise ValueError('Auto cannot release this model yet: a downstream connection retains an opaque or device-backed output.')


def release_owner_caches(server, event, memory_manager):
    import sys
    import weakref
    targets = set(event['nodeIds'])
    manager = getattr(sys.modules.get('modules.ModularDiffusers'), 'components', None)
    collections = getattr(manager, 'collections', {})
    entries = getattr(manager, 'components', {})
    selected = {key for owner in targets for key in collections.get(owner, [])}
    shared = {key for owner, keys in collections.items() if owner not in targets for key in keys}
    component_refs = {key: weakref.ref(entries[key]) for key in selected - shared if key in entries and hasattr(entries[key], 'parameters')}
    before = server._cuda_memory_snapshot() if hasattr(server, '_cuda_memory_snapshot') else None
    preserved = {}
    # Preflight every retained output before any destructive mutation.
    for node_id, fields in event['retainOutputs'].items():
        cached = server.node_cache.get(node_id)
        if cached is None or not isinstance(cached.output, dict) or any(field not in cached.output for field in fields):
            raise ValueError('Auto cannot release an owner with a missing downstream output.')
        preserved[node_id] = {field: _material(cached.output[field]) for field in fields}
    survivors = [node for node_id, node in server.node_cache.items() if node_id not in targets]
    shared_ids = {model_id for node in survivors for model_id in getattr(node, '_mm_models', [])}
    released_ids = {model_id for node_id in targets for model_id in getattr(server.node_cache.get(node_id), '_mm_models', [])} - shared_ids
    # Destruction, not CPU offload: remove ownership first, exactly like the
    # established idle turnover path, avoiding a transient full CPU weight copy.
    for model_id in released_ids:
        memory_manager.cache.pop(model_id, None)
    released_components = server._release_node_modular_components(targets)
    for node_id in targets:
        cached_node = server.node_cache.pop(node_id, None)
        if cached_node is not None:
            # Manager ownership has already been pruned. Destructors must not
            # remove a model ID still shared by an unexpired node.
            cached_node._mm_models = []
    cached_node = None
    cached = None
    for node_id, output in preserved.items():
        server.node_cache[node_id] = SimpleNamespace(output=output, params={}, _has_changed=True, _mm_models=[])
    import gc
    gc.collect()
    errors = server._best_effort_device_cache_clear()
    server._best_effort_allocator_trim()
    if errors:
        raise ValueError('Auto released model references but could not clear the device allocator: ' + '; '.join(errors))
    remaining = [key for key, reference in component_refs.items() if reference() is not None]
    after = server._cuda_memory_snapshot() if hasattr(server, '_cuda_memory_snapshot') else None
    import logging
    logging.getLogger('modiff').info(f'Auto owner release: {released_components} components; remaining references {remaining}; memory {before} -> {after}')
    if remaining:
        raise ValueError('Auto stopped after releasing an owner: model references remain alive: ' + ', '.join(remaining))
    return {'ownerIds': event['ownerIds'], 'releasedNodeIds': sorted(targets), 'retainedOutputNodes': sorted(preserved),
            'releasedComponentCount': released_components, 'memoryBefore': before, 'memoryAfter': after}


def assert_next_owner_capacity(server, owner, hardware=None):
    """Check actual free memory after releases, without adding reclaimable cache."""
    from modiff.auto_resource import _hardware_snapshot
    snapshot = hardware if hardware is not None else _hardware_snapshot(server._runtime_fingerprint(), server.data_dir)
    accelerator = snapshot.get('accelerator', {})
    ram = snapshot.get('systemMemory', {}).get('availableBytes')
    vram = accelerator.get('freeBytes')
    shared = accelerator.get('memoryKind') in {'shared', 'unified', 'shared_system', 'system_shared'} or accelerator.get('sharedMemory') is True
    if shared:
        devices = snapshot.get('runtime', {}).get('hardware', {}).get('devices', [])
        device = next((item for item in devices if item.get('type') == accelerator.get('kind') and item.get('memory_kind') in {'shared', 'unified'}), {})
        values = [value for value in (device.get('shared_memory_free'), device.get('torch_vram_free'), ram) if isinstance(value, (int, float))]
        vram = min(values) if values else vram
    requirements = owner['requirements']
    demand = requirements['systemRamBytes'] + (requirements['vramBytes'] if shared else 0)
    if (demand and (ram is None or demand > ram)) or (requirements['vramBytes'] and (vram is None or requirements['vramBytes'] > vram)):
        raise ValueError(f"Auto stopped before loading {owner['ownerId']}: actual free memory after the release is below its planned requirement.")
