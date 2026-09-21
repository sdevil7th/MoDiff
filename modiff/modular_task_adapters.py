"""Task projections shared by operation discovery and executable graph inspection."""

from modiff.huggingface_node_library import graph_adapter_contracts
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH


def modular_task_adapters(pipeline, workflow):
    adapters = graph_adapter_contracts(pipeline, workflow["id"], workflow)
    whole = reviewed_whole_workflow_graph_adapter(pipeline, workflow["id"])
    if whole:
        truth = PINNED_MODULAR_WORKFLOW_TRUTH.get(pipeline)
        tasks = (
            [mode for mode, route in truth.modes if (route.upstream_workflow or "default") == workflow["id"]]
            if truth
            else []
        )
        return [(task, whole) for task in tasks or [workflow["taskId"]]]
    result = {}
    for adapter in adapters:
        if not adapter["actionSequence"] or "full_pipeline" in adapter["actionSequence"]:
            continue  # A whole standard call does not establish Modular stages.
        task = adapter["adapterId"] if adapter["source"] == "mode" else workflow["taskId"]
        result.setdefault(task, adapter)
    return list(result.items())
