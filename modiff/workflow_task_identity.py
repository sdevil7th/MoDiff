"""Read-only task recognition from existing reviewed Modular graph contracts.

Authoring labels are not execution authority. Match concrete node identities and
state wires; unknown, partial and competing compositions remain unresolved.
This does not qualify a model or a resource recipe, or modify the submitted graph.
"""

from collections import Counter
import re

from modiff.modular_action_bindings import MODULAR_ACTION_BINDINGS
from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH


def modular_graph_tasks(nodes, loader_id, consumer_ids, pipeline_class):
    truth = PINNED_MODULAR_WORKFLOW_TRUTH.get(pipeline_class)
    if truth is None:
        return set()
    stages = {
        key: nodes[key]
        for key in consumer_ids
        if key != loader_id and key in nodes and nodes[key].get("module") == "modules.ModularDiffusers"
    }
    identities = Counter(f"{n['module']}.{n['action']}" for n in stages.values())
    candidates = []
    for task, route in truth.modes:
        names = {
            action: MODULAR_ACTION_BINDINGS[action][1]
            for action in route.action_sequence
            if action in MODULAR_ACTION_BINDINGS
        }
        if len(names) != len(route.action_sequence) or Counter(names.values()) != identities:
            continue
        if any(count != 1 for count in identities.values()):
            continue  # Several branches cannot be labeled as one operation chain.
        ids = {
            action: next(key for key, node in stages.items() if f"{node['module']}.{node['action']}" == name)
            for action, name in names.items()
        }
        expected = {
            (ids[e.producer_action], e.producer_output, ids[e.consumer_action], e.consumer_input)
            for e in route.state_edges
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
        if actual == expected:
            candidates.append((task, route))
    # Some tasks share actions/wires but differ in optional conditioning inputs.
    # Only discriminate the fields declared by those exact candidate contracts.
    discriminators = set().union(*(r.required_upstream_inputs for _, r in candidates)) if candidates else set()
    common = set.intersection(*(set(r.required_upstream_inputs) for _, r in candidates)) if candidates else set()
    discriminators -= common
    supplied = {
        field
        for node in stages.values()
        for field, param in node.get("params", {}).items()
        if field in discriminators
        and isinstance(param, dict)
        and (param.get("sourceId") or param.get("value") not in (None, "", [], {}))
    }
    return {task for task, route in candidates if set(route.required_upstream_inputs) & discriminators == supplied}


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
            if p.get("sourceId") and "loop_member" in str(p.get("sourceKey", "")) and p["sourceId"] not in found
        )
        if not additions:
            return sorted(found - {loader_id})
        found.update(additions)


def graph_task_receipts(nodes, records):
    """Recognize only the output ancestry with captured Modular loader identity."""
    loaders = {
        key
        for key, node in nodes.items()
        if (node.get("module"), node.get("action")) == ("modules.ModularDiffusers", "ModelsLoader")
    }
    result = []
    for key in sorted(loaders):
        model = records.get(key, {}).get("fields", {}).get("model_type", {}).get("value")
        if not isinstance(model, str):
            return []  # Incomplete owner evidence must not select another owner's task.
        tasks = modular_graph_tasks(nodes, key, resource_consumers(nodes, key, loaders), model)
        result.append(
            {"loaderId": key, "pipelineClass": model, "task": next(iter(tasks)) if len(tasks) == 1 else None}
        )
    return result
