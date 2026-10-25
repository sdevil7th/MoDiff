"""Read-only task recognition from existing reviewed Modular graph contracts.

Authoring labels are not execution authority. Match concrete node identities and
state wires; unknown, partial and competing compositions remain unresolved.
This does not qualify a model or a resource recipe, or modify the submitted graph.
"""

from collections import Counter
import json
import re

from modiff.modular_action_bindings import MODULAR_ACTION_BINDINGS, MODULAR_AUXILIARY_OPERATION_BINDINGS
from modiff.modular_task_adapters import modular_task_adapters
from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot


def _composition_tasks(stages, pipeline_class, workflows):
    """Recognize the pinned selected workflow of a validated explicit composition.

    Placement validation is the runtime's read-only validator. It grants neither
    executable-code trust nor a resource recipe. Partial non-media scopes stay
    unlabeled, as do mixed workflow owners or invalid imported selectors.
    """
    from modiff.modular_composition import validate_modular_composition_recipe
    from modiff.huggingface_node_library import reviewed_huggingface_node_library
    from modiff.modular_conditional_contracts import reviewed_modular_conditional_snapshot
    from modules.ModularDiffusers.reviewed_blocks import _reviewed_placement

    selected = set()
    media = False
    library = reviewed_huggingface_node_library()
    snapshot = None
    compositions = {}
    for node in stages.values():
        if node['action'] != 'ReviewedModularWorkflowStep':
            return set()
        values = {k: p.get('value') for k, p in node.get('params', {}).items() if isinstance(p, dict)}
        if values.get('pipeline_class') != pipeline_class:
            return set()
        try:
            recipe = values.get('composition_recipe')
            composition = None
            if recipe is not None:
                key = json.dumps(recipe, sort_keys=True, allow_nan=False)
                if key not in compositions:
                    compositions[key] = validate_modular_composition_recipe(recipe, library=library)
                composition = compositions[key]
            if snapshot is None and (composition is not None or values.get('execution_scope') == 'unpruned_pipeline'
                                     or values.get('workflow_id') == 'default'):
                snapshot = reviewed_modular_conditional_snapshot()
            _, block = _reviewed_placement(
                pipeline_class=pipeline_class, workflow_id=values.get('workflow_id'),
                execution_scope=values.get('execution_scope') or 'selected_workflow',
                placement_path=tuple(values.get('placement_path') or ()),
                block_definition_id=values.get('block_definition_id'), block_class=values.get('block_class'),
                block_hash=values.get('block_contract_hash'), composition=composition, library=library, snapshot=snapshot,
            )
        except (ValueError, TypeError, KeyError):
            return set()
        selected.add(values.get('workflow_id'))
        media |= any(p['name'] in {'images', 'videos', 'audio', 'audios', 'sound'} for p in block['outputs'])
    if len(selected) != 1 or not media:
        return set()
    return {task for workflow in workflows if workflow['id'] in selected
            for task, _ in modular_task_adapters(pipeline_class, workflow)}


def modular_graph_tasks(nodes, loader_id, consumer_ids, pipeline_class):
    workflows = next((p['workflows'] for p in load_reviewed_modular_workflow_snapshot()['contracts']
                      if p['pipelineClass'] == pipeline_class), ())
    if not workflows:
        return set()
    helpers = {binding[1] for binding in MODULAR_AUXILIARY_OPERATION_BINDINGS.values()}
    helpers.update({"modules.ModularDiffusers.Guider", "modules.ModularDiffusers.Layers"})
    stages = {
        key: nodes[key]
        for key in consumer_ids
        if key != loader_id and key in nodes and nodes[key].get("module") == "modules.ModularDiffusers"
        and f"{nodes[key]['module']}.{nodes[key]['action']}" not in helpers
    }
    if any(n['action'] == 'ReviewedModularWorkflowStep' for n in stages.values()):
        return _composition_tasks(stages, pipeline_class, workflows)
    identities = Counter(f"{n['module']}.{n['action']}" for n in stages.values())
    candidates = []
    routes = [(task, adapter, workflow['id']) for workflow in workflows
              for task, adapter in modular_task_adapters(pipeline_class, workflow)]
    for task, route, workflow_id in routes:
        names = {
            action: MODULAR_ACTION_BINDINGS[action][1]
            for action in route["actionSequence"]
            if action in MODULAR_ACTION_BINDINGS
        }
        if len(names) != len(route["actionSequence"]) or Counter(names.values()) != identities:
            continue
        if any(count != 1 for count in identities.values()):
            continue  # Several branches cannot be labeled as one operation chain.
        ids = {
            action: next(key for key, node in stages.items() if f"{node['module']}.{node['action']}" == name)
            for action, name in names.items()
        }
        # Whole-workflow stages bind exact executable selectors. A partial or
        # conflicting selection is not rescued by advisory authoring metadata.
        if any(node.get('params', {}).get('pipeline_class', {}).get('value') != pipeline_class
               or node.get('params', {}).get('workflow_id', {}).get('value') != workflow_id
               for node in stages.values() if node['action'].startswith('Workflow')):
            continue
        selected_workflow = nodes.get(loader_id, {}).get('params', {}).get('workflow_id', {}).get('value')
        if selected_workflow and selected_workflow != workflow_id:
            continue
        expected = {
            (ids[e['producerAction']], e['producerOutput'], ids[e['consumerAction']], e['consumerInput'])
            for e in route['stateEdges']
        }
        actual = {
            (param["sourceId"], param.get("sourceKey"), key, field)
            for key, node in stages.items()
            for field, param in node.get("params", {}).items()
            if isinstance(param, dict) and param.get("sourceId") in stages
        }
        # Effective dimensions are ordinary values emitted by Denoise, not
        # task-selecting state links. They are explicitly wired by starters.
        actual = {
            edge for edge in actual if not (edge[1], edge[3]) in {("out_width", "width"), ("out_height", "height")}
        }
        if actual != expected:
            # Starters also wire complete source-declared wrapper transports.
            # Accept that exact bundle, never an arbitrary extra stage link.
            from modiff.studio_execution_specs import studio_execution_spec_for_pair

            profile = nodes.get(loader_id, {}).get('params', {}).get('execution_profile_id', {}).get('value')
            specification = studio_execution_spec_for_pair(
                pipeline_class, task, execution_profile_id=profile,
            )
            if specification is not None:
                by_key = {f"{node['module']}.{node['action']}": key for key, node in stages.items()}
                roles = {role: by_key[key] for role, key, *_ in specification['roles'] if key in by_key}
                expected |= {
                    (roles[source], output, roles[target], field)
                    for source, output, target, field in specification['edges']
                    if source in roles and target in roles
                    and (output, field) not in {('out_width', 'width'), ('out_height', 'height')}
                }
        if actual == expected:
            candidates.append((task, route))
    # Some tasks share actions/wires but differ in optional conditioning inputs.
    # Only discriminate the fields declared by those exact candidate contracts.
    discriminators = set().union(*(set(r["requiredInputs"]) for _, r in candidates)) if candidates else set()
    common = set.intersection(*(set(r["requiredInputs"]) for _, r in candidates)) if candidates else set()
    discriminators -= common
    # Numeric defaults (for example a control selector at zero) do not prove
    # which optional branch a generic node selected. Retain that ambiguity.
    discriminators -= {
        field for node in [nodes.get(loader_id, {}), *stages.values()]
        for field, param in node.get("params", {}).items()
        if isinstance(param, dict) and not param.get("sourceId")
        and isinstance(param.get("value"), (int, float, bool))
    }
    supplied = {
        field
        for node in [nodes.get(loader_id, {}), *stages.values()]
        for field, param in node.get("params", {}).items()
        if field in discriminators
        and isinstance(param, dict)
        and (param.get("sourceId") or (not isinstance(param.get("value"), (int, float, bool))
                                     and param.get("value") not in (None, "", [], {})))
    }
    return {task for task, route in candidates if set(route["requiredInputs"]) & discriminators == supplied}


_RESOURCE_LINK = re.compile(r"pipeline|component|state|model|encoder|unet|vae|loop_member", re.I)


def resource_consumers(nodes: dict, loader_id: str, loader_ids: set[str]) -> list[str]:
    found = {loader_id}
    while True:
        additions = {
            node_id
            for node_id, node in nodes.items()
            if node_id not in found | loader_ids
            and any(
                p.get("sourceId") in found and _RESOURCE_LINK.search(str(p.get("sourceKey", "")) + " " + key)
                for key, p in node["params"].items()
            )
        }
        additions.update(
            p["sourceId"]
            for node_id in found
            for key, p in nodes[node_id]["params"].items()
            if p.get("sourceId") in nodes and p["sourceId"] not in found | loader_ids
            # Follow upstream component/state suppliers, not images produced by
            # a completed model owner. Crossing a media edge would merge owners
            # and prevent the existing sequential resource-release schedule.
            and nodes[p["sourceId"]].get("module") == "modules.ModularDiffusers"
            and (
                re.search(r"pipeline|component|state|model|encoder|unet|vae|loop_member|controlnet|adapter",
                          str(p.get("sourceKey", "")) + " " + key, re.I)
                # Native guidance helpers supply configuration upstream of the
                # denoiser. They own no weights but belong to this model scope.
                or (nodes[p["sourceId"]].get("action"), p.get("sourceKey"), key) in {
                    ("Guider", "guider_out", "guider"),
                    ("Layers", "layers_config", "layers_config"),
                }
            )
        )
        if not additions:
            return sorted(found - {loader_id})
        found.update(additions)


def direct_image_stage_owner(nodes: dict, stage_id: str) -> tuple[str, tuple[str, ...]]:
    """Resolve one exclusive builtin pipeline/adapter route without execution.

    This proves topology only, never adapter bytes, placement or warm storage.
    Each intermediate output must feed the next adapter alone; the terminal
    output must feed exactly the three distinct current image stage actions.
    """
    stages = {'EncodeInputs', 'Denoise', 'DecodeLatents'}
    node = nodes.get(stage_id, {})
    if node.get('module') != 'modules.DiffusersImage' or node.get('action') not in stages:
        raise ValueError('A declared direct image stage is required.')
    pipeline = node.get('params', {}).get('pipeline', {})
    endpoint = (pipeline.get('sourceId'), pipeline.get('sourceKey'))
    current, output = endpoint
    reverse, seen = [], set()
    while current:
        if current in seen or len(reverse) > 32:
            raise ValueError('The direct image adapter route is cyclic or exceeds 32 adapters.')
        seen.add(current)
        source = nodes.get(current, {})
        identity = (source.get('module'), source.get('action'))
        if identity == ('modules.DiffusersImage', 'LoadPipeline') and output == 'pipeline':
            loader = current
            break
        if identity != ('modules.DiffusersImage', 'LoadAdapter') or output != 'output':
            raise ValueError('The image stage needs a declared loader or ordered adapter output.')
        reverse.append(current)
        upstream = source.get('params', {}).get('pipeline', {})
        current, output = upstream.get('sourceId'), upstream.get('sourceKey')
    else:
        raise ValueError('The image stage has no concrete pipeline owner.')
    adapters = tuple(reversed(reverse))
    links = [(loader, 'pipeline'), *((key, 'output') for key in adapters)]
    for index, (source_id, source_key) in enumerate(links):
        uses = [(key, field) for key, value in nodes.items()
                for field, param in value['params'].items()
                if (param.get('sourceId'), param.get('sourceKey')) == (source_id, source_key)]
        if index < len(adapters):
            if uses != [(adapters[index], 'pipeline')]:
                raise ValueError('An intermediate image adapter output is shared or bypassed.')
        elif (len(uses) != 3 or {field for _, field in uses} != {'pipeline'}
              or {nodes[key]['action'] for key, _ in uses} != stages
              or any(nodes[key]['module'] != 'modules.DiffusersImage' for key, _ in uses)
              or stage_id not in {key for key, _ in uses}):
            raise ValueError('The terminal image owner output needs its sole complete stage trio.')
    return loader, adapters


_IMAGE_OWNERS = {
    ('modules.DiffusersImage', 'LoadPipeline'),
    ('modules.HuggingFaceTransformers', 'LoadDepthEstimationModel'),
    ('modules.HuggingFaceTransformers', 'LoadImageTextToTextModel'),
    ('modules.ImageOperations', 'ProcessImage'),
}


def _ordinary_image_tasks(nodes, records, key, loaders, model):
    module, action = nodes[key]['module'], nodes[key]['action']
    if module == 'modules.ImageOperations':
        from modules.ImageOperations.main import IMAGE_OPERATION_MODES

        actions = {action: set(IMAGE_OPERATION_MODES)} if model == 'BuiltinImageOperationV1' else {}
        consumers = [key]
    elif module == 'modules.DiffusersImage':
        from modules.DiffusersImage.main import IMAGE_ACTION_MODES, IMAGE_PIPELINE_ADAPTERS, image_pipeline_contract

        adapter = IMAGE_PIPELINE_ADAPTERS.get(model)
        actions = {a: set(tasks) & set(adapter.modes) for a, tasks in IMAGE_ACTION_MODES.items()} if adapter else {}
        stage_actions = {'EncodeInputs', 'Denoise', 'DecodeLatents'}
        owner_task = records.get(key, {}).get('fields', {}).get('mode', {}).get('value')
        if adapter and adapter.stage_adapter is not None and isinstance(owner_task, str) and owner_task in adapter.modes:
            declared = image_pipeline_contract(adapter, owner_task)['actions']
            actions.update({action: set(declared[action]) for action in stage_actions if action in declared})
        consumers = resource_consumers(nodes, key, loaders)
    else:
        actions = {
            'PredictDepth': {'depth_estimation'} if model == 'AutoModelForDepthEstimation' else set(),
            'GenerateImageVideoText': {'image_to_text'} if model == 'AutoModelForImageTextToText' else set(),
        }
        consumers = resource_consumers(nodes, key, loaders)
    tasks = set()
    for consumer in consumers:
        node = nodes[consumer]
        if node['module'] != module:
            continue
        if module == 'modules.DiffusersImage' and node['action'] in stage_actions and node['action'] not in actions:
            return set()  # Missing owner capture/undeclared stage is not an image-task guess.
        if node['action'] not in actions:
            continue
        task = records.get(consumer, {}).get('fields', {}).get('executed_task', {}).get('value')
        if not isinstance(task, str) or task not in actions[node['action']]:
            return set()
        tasks.add(task)
    return tasks


def graph_task_receipts(nodes, records):
    """Use captured owner identity and validated image calls in output ancestry."""
    loaders = {
        key
        for key, node in nodes.items()
        if (node.get("module"), node.get("action")) in _IMAGE_OWNERS | {('modules.ModularDiffusers', 'ModelsLoader')}
    }
    result = []
    for key in sorted(loaders):
        modular = nodes[key]['module'] == 'modules.ModularDiffusers'
        model = records.get(key, {}).get("fields", {}).get("model_type" if modular else 'pipeline_class', {}).get("value")
        if not isinstance(model, str):
            return []  # Incomplete owner evidence must not select another owner's task.
        tasks = (modular_graph_tasks(nodes, key, resource_consumers(nodes, key, loaders), model) if modular
                 else _ordinary_image_tasks(nodes, records, key, loaders, model))
        result.append(
            {"loaderId": key, "pipelineClass": model, "task": next(iter(tasks)) if len(tasks) == 1 else None}
        )
    return result
