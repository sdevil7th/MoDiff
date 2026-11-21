"""Private native image disk-LoRA loading and exact owner mutation identity."""
from __future__ import annotations

from collections.abc import Mapping
import sys
import weakref

from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.native_image_auto_placement import declared_image_owner


def declared_lora_components(model_type: str, repository: str) -> tuple[str, ...] | None:
    """Read reviewed targets from the actual loader's canonical identity."""
    if (not isinstance(model_type, str) or model_type != model_type.strip() or not model_type
            or not isinstance(repository, str) or repository != repository.strip() or not repository):
        return None
    native_profiles = [profile for profile in DIFFUSERS_EXECUTION_PROFILES.values()
                       if profile.execution_path == 'modular-diffusers'
                       and profile.backend_path == 'modules.ModularDiffusers.ModelsLoader']
    canonical_profiles = [profile for profile in native_profiles if profile.pipeline_class == model_type]
    if canonical_profiles:
        from modiff.diffusers_profiles import resolve_execution_profiles_for_loader

        resolved, reason = resolve_execution_profiles_for_loader(
            'modules.ModularDiffusers', 'ModelsLoader',
            {'model_type': model_type, 'repo_id': {'source': 'hub', 'value': repository}},
        )
        profiles = [profile for profile in resolved
                    if profile.execution_path == 'modular-diffusers'
                    and profile.backend_path == 'modules.ModularDiffusers.ModelsLoader'
                    and profile.pipeline_class == model_type
                    and repository in {profile.default_repo, profile.fallback_repo, *profile.compatible_repos}
                    and any(declared_image_owner(profile, repository, mode) for mode in profile.modes)]
        # The resolver's sole-profile shortcut does not verify repository membership.
        # Never use logical aliases to bypass a conflicting canonical selection.
        if reason is not None or len(profiles) != 1:
            return () if profiles else None
    else:
        # Preserve declared logical alias callers only when no canonical class exists.
        profiles = [profile for profile in native_profiles if profile.model_type == model_type
                    and repository in {profile.default_repo, profile.fallback_repo, *profile.compatible_repos}
                    and any(declared_image_owner(profile, repository, mode) for mode in profile.modes)]
    targets = {profile.lora_loadable_components for profile in profiles}
    return targets.pop() if len(targets) == 1 else () if profiles else None


def requested_disk_lora_policy(settings: Mapping, *, component_context=None) -> tuple | None:
    """Resolve descriptors only at normal dispatch; inspection uses static targets."""
    if component_context is not None:
        from modiff.native_component_lora import component_policy
        return component_policy(component_context, settings)
    if (settings.get('offload_mode', settings.get('offloadMode')) != 'group_disk'
            or settings.get('auto_offload', True) is False
            or str(settings.get('device', '')).startswith('cpu')):
        return None
    model_type = settings.get('model_type', settings.get('modelType'))
    repository = settings.get('repo_id', settings.get('modelRepo'))
    if isinstance(repository, Mapping):
        repository = repository.get('value')
    targets = declared_lora_components(model_type, repository)
    if targets is None:
        return None
    loras = settings.get('lora_list')
    if not targets:
        if loras:
            raise ValueError('Native image disk LoRA requires reviewed immutable component loading targets.')
        return None
    from modiff.auxiliary_lora import resolve_lora_descriptors, scheduler_override_contract

    resolved = resolve_lora_descriptors(loras) if loras is not None and loras != [] else []
    scheduler_override_contract(resolved)
    return (targets, tuple(item.descriptor_sha256 for item in resolved))


def graph_disk_lora_policy(settings: Mapping, graph, node_id, *, component_context=None):
    """Compare already admitted literal native descriptors before Auto replans."""
    base = requested_disk_lora_policy({**settings, 'lora_list': None}, component_context=component_context)
    if base is None:
        return None
    param = graph['nodes'][node_id].get('params', {}).get('lora_list', {})
    terminal = param.get('sourceId')
    if terminal is None:
        return requested_disk_lora_policy({**settings, 'lora_list': param.get('value')}, component_context=component_context)
    from modiff.auxiliary_lora import controlled_lora_receipts_from_graph, native_lora_chain_ids

    if param.get('sourceKey') != 'lora':
        raise ValueError('Native image disk LoRA requires the declared ordered lora output.')
    chain = native_lora_chain_ids(graph, terminal)
    receipts = controlled_lora_receipts_from_graph({
        'nodes': {key: graph['nodes'][key] for key in chain}, 'paths': [chain],
    })
    if component_context is not None:
        from modiff.native_component_lora import component_chain_policy
        return component_chain_policy(component_context, graph, chain, receipts)
    return (base[0], tuple(receipt['descriptorSha256'] for receipt in receipts), *base[2:])


def validate_disk_lora_targets(pipeline, policy, selected_components, manager, node_id):
    """Refuse missing/drifted targets and shared source-owned mutation inputs."""
    if policy is None:
        return ()
    targets = policy[0]
    from modiff.native_component_lora import is_component_policy
    if is_component_policy(policy):
        if any(name in selected_components for name in targets):
            raise ValueError('Selected component LoRA targets must be fresh and exclusively owned.')
        return targets
    actual = getattr(type(pipeline), '_lora_loadable_modules', None)
    if not isinstance(actual, (list, tuple)) or tuple(actual) != targets:
        raise ValueError('Native image disk LoRA targets differ from the reviewed installed pipeline contract.')
    if any(name not in pipeline.pretrained_component_names for name in targets):
        raise ValueError('Native image disk LoRA requires every declared target component.')
    for name in targets:
        if name in selected_components:
            component = selected_components[name]
            if (getattr(component, '_modiff_offload_node_id', None) != str(node_id)
                    or getattr(component, '_modiff_disk_lora_policy', None) != policy):
                raise ValueError('Connected disk LoRA targets must remain exclusively owned by their source loader.')
            _assert_exclusive_target(manager, node_id, component)
    return targets


def _assert_exclusive_target(manager, node_id, component):
    ids = {key for key, value in manager.components.items() if value is component}
    if (not ids or not ids.issubset(manager.collections.get(node_id, set()))
            or any(ids.intersection(values) for owner, values in manager.collections.items() if owner != node_id)):
        raise ValueError('Native disk LoRA targets cannot mutate components shared by another owner.')


def publish_disk_lora_binding(node, policy, manager):
    """Publish only after official loading, activation, scheduler and hooks succeed."""
    if policy is None:
        return
    pipeline = node.loader
    bound = []
    for name, component in pipeline.components.items():
        if name in policy[0]:
            _assert_exclusive_target(manager, node.node_id, component)
        ids = sorted(key for key, value in manager.components.items() if value is component)
        if not ids:
            continue
        try:
            reference = weakref.ref(component)
        except TypeError:
            return  # Opaque registered metadata cannot attest complete reuse.
        bound.append((name, tuple(ids), reference))
    for name in policy[0]:
        pipeline.components[name]._modiff_disk_lora_policy = policy
    node._native_disk_lora_state = {'policy': policy, 'owner': node.node_id,
                                  'pipeline': weakref.ref(pipeline), 'components': tuple(bound)}
    from modiff.native_component_lora import is_component_policy, pending_component_binding
    if is_component_policy(policy):
        pending_component_binding(node, node._native_disk_lora_state, node._selected_component_lora_context)


def disk_lora_binding_matches(node, policy, manager) -> bool:
    """An old marker never blesses an evicted/replaced/partial owner incarnation."""
    from modiff.node_cache_identity import implementation_identity

    state = getattr(node, '_native_disk_lora_state', None)
    if (not isinstance(state, dict) or state.get('policy') != policy or state.get('owner') != node.node_id
            or not getattr(node, '_cache_valid', False) or getattr(node, '_cache_invalidated', False)
            or state['pipeline']() is not getattr(node, 'loader', None)
            or state.get('outputIdentity') != id(getattr(node, 'output', None))
            or state.get('implementation') != implementation_identity(node)):
        return False
    from modiff.native_component_lora import component_binding_matches
    if not component_binding_matches(node, state):
        return False
    for name, ids, reference in state['components']:
        component = reference()
        if (component is None or getattr(node.loader, name, None) is not component
                or any(manager.components.get(key) is not component
                       or key not in manager.collections.get(node.node_id, set()) for key in ids)
                or name in policy[0] and getattr(component, '_modiff_disk_lora_policy', None) != policy):
            return False
        if name in policy[0]:
            try:
                _assert_exclusive_target(manager, node.node_id, component)
            except ValueError:
                return False
    return set(policy[0]).issubset(name for name, _, _ in state['components'])


def record_disk_lora_output(node, args):
    """Bind successful ordinary dispatch, after NodeBase has published output."""
    loader_module = sys.modules.get('modules.ModularDiffusers.loaders')
    observed_owner = loader_module is not None and type(node) is vars(loader_module).get('ModelsLoader')
    try:
        state = getattr(node, '_native_disk_lora_state', None)
        if state:
            from modiff.node_cache_identity import implementation_identity
            state['dispatchInputs'] = dispatch_inputs(args)
            state['outputIdentity'] = id(node.output)
            state['implementation'] = implementation_identity(node)
            from modiff.native_component_lora import complete_component_binding
            complete_component_binding(node, state)
        if observed_owner:
            from modiff.applied_adapter_provenance import seal_adapter_observation
            from modules.ModularDiffusers.loaders import components
            seal_adapter_observation(node, components)
    except BaseException:
        if observed_owner:
            from modiff.applied_adapter_provenance import abort_adapter_observation
            abort_adapter_observation(node)
        raise


def dispatch_inputs(args):
    """Snapshot ordinary submitted/resolved inputs; internal preflight tokens vary."""
    from modiff.node_cache_identity import input_snapshot
    return input_snapshot({key: value for key, value in args.items() if not key.startswith('_')})


def graph_dispatch_inputs(graph, node_id, cache, prepared_nodes=()):
    """Read existing supplier outputs only; unknown values require cold teardown."""
    from modiff.node_cache_identity import implementation_identity

    args = {}
    changed = False
    for key, param in graph['nodes'][node_id].get('params', {}).items():
        if param.get('display') in {'ui_group', 'ui_text', 'ui_image', 'ui_imagecompare', 'ui_areaselect',
                                    'ui_audio', 'ui_video', 'ui_3d', 'ui_label', 'ui_button'}:
            continue
        source = param.get('sourceId')
        if source:
            node = cache.get(source)
            output = getattr(node, 'output', None)
            if not isinstance(output, dict) or param.get('sourceKey') not in output:
                return None, True
            args[key] = output[param['sourceKey']]
            # _has_changed persists from the PREVIOUS call until the next call.
            # Submitted upstream inputs are already bound by the Auto owner key;
            # only this attempt's data preparation may use that flag here.
            changed = changed or (not getattr(node, '_cache_valid', False)
                or getattr(node, '_cache_invalidated', False)
                or getattr(node, '_cache_implementation', None) != implementation_identity(node)
                or source in prepared_nodes and bool(getattr(node, '_has_changed', False)))
        else:
            args[key] = param.get('value')
    return dispatch_inputs(args), changed
