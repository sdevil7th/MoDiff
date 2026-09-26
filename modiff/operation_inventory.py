"""Pinned Diffusers export/task inventory; discovery reads the checked artifact.

AutoPipeline mappings describe dispatch tasks, not interchangeable model inputs.
Exports without named Auto/Modular tasks retain their direct pipeline-call surface
and the existing explicit review decision. No model imports are needed to rebuild.
"""

import ast
import hashlib
import json
from pathlib import Path

from modiff.operation_contracts import _identifier
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot

OPERATION_INVENTORY_PATH = Path(__file__).resolve().parents[1] / "data" / "diffusers-operation-inventory.v1.json"
_AUTO_TASKS = {
    "TEXT2IMAGE": "text_to_image",
    "IMAGE2IMAGE": "image_to_image",
    "INPAINT": "inpaint",
    "TEXT2VIDEO": "text_to_video",
    "IMAGE2VIDEO": "image_to_video",
    "VIDEO2VIDEO": "video_to_video",
    "CONDITION2VIDEO": "condition_to_video",
    "TEXT2AUDIO": "text_to_audio",
}


def _hash(value):
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _auto_tasks(source):
    """Read all declared entries, including conditional dependency registrations."""
    result = {}
    names = {}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.endswith("_PIPELINES_MAPPING"):
            name = target.id
            task_key = (
                name.removeprefix("_")
                .removeprefix("AUTO_")
                .removesuffix("_PIPELINES_MAPPING")
                .removesuffix("_DECODER")
            )
            if task_key not in _AUTO_TASKS:
                raise ValueError(f"Unreviewed AutoPipeline task mapping: {name}")
            names[name] = _AUTO_TASKS[task_key]
            value = node.value
            if (
                not isinstance(value, ast.Call)
                or not isinstance(value.func, ast.Name)
                or value.func.id != "OrderedDict"
            ):
                raise ValueError("Unsupported AutoPipeline mapping declaration.")
            for entry in value.args[0].elts:
                pipeline = entry.elts[1]
                if not isinstance(pipeline, ast.Name):
                    raise ValueError("Unsupported AutoPipeline class reference.")
                result.setdefault(pipeline.id, set()).add(names[name])
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in names:
                if not isinstance(node.value, ast.Name):
                    raise ValueError("Unsupported conditional AutoPipeline class reference.")
                result.setdefault(node.value.id, set()).add(names[target.value.id])
    return result


def build_operation_inventory(root: Path, *, diffusers_source=None):
    # Source auditing stays out of ordinary startup/discovery.
    from modiff.upstream_coverage import (
        _pipeline_coverage,
        _verify_diffusers_source_revision,
        installed_diffusers_source,
    )

    source = diffusers_source or installed_diffusers_source()
    _verify_diffusers_source_revision(source, PINNED_DIFFUSERS_REVISION)
    version, coverage = _pipeline_coverage(root, source)
    auto_path = source / "pipelines" / "auto_pipeline.py"
    auto = _auto_tasks(auto_path.read_text())
    snapshot_path = root / "data" / "modular-workflow-contracts.json"
    snapshot = load_reviewed_modular_workflow_snapshot(snapshot_path)
    modular = {c["pipelineClass"]: c for c in snapshot["contracts"]}
    symbols = {p["name"] for p in coverage}
    if set(auto) - symbols or set(modular) - symbols:
        raise ValueError("Task inventory refers to an unexported pipeline.")
    pipelines = []
    for item in coverage:
        name = item["name"]
        tasks = [{"task": task, "source": "auto", "workflowId": None} for task in sorted(auto.get(name, ()))]
        tasks.extend(
            {"task": w["taskId"], "source": "modular", "workflowId": w["id"]}
            for w in modular.get(name, {}).get("workflows", [])
        )
        if not tasks:
            tasks = [{"task": "pipeline_call", "source": "export", "workflowId": None}]
        pipelines.append(
            {
                "pipelineClass": name,
                "coverage": item["status"],
                "reason": item["reason"],
                "equivalentTo": item["equivalentTo"],
                "upstreamTasks": tasks,
            }
        )
    result = {
        "schemaVersion": 1,
        "diffusersRevision": PINNED_DIFFUSERS_REVISION,
        "diffusersVersion": version,
        "sources": {
            name: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("exports", source / "__init__.py"),
                ("autoTasks", auto_path),
                ("modularTasks", snapshot_path),
            )
        },
        "pipelines": pipelines,
    }
    return {**result, "contentHash": _hash(result)}


def load_operation_inventory(path=OPERATION_INVENTORY_PATH):
    raw = path.read_bytes()
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Operation inventory is oversized.")
    value = json.loads(raw)
    semantic = {key: item for key, item in value.items() if key != "contentHash"}
    if (
        set(value) != {"schemaVersion", "diffusersRevision", "diffusersVersion", "sources", "pipelines", "contentHash"}
        or type(value.get("schemaVersion")) is not int
        or value.get("schemaVersion") != 1
        or value.get("diffusersRevision") != PINNED_DIFFUSERS_REVISION
        or value.get("contentHash") != _hash(semantic)
    ):
        raise ValueError("Operation inventory does not match the reviewed source contract.")
    if not isinstance(value["pipelines"], list) or not 1 <= len(value["pipelines"]) <= 512:
        raise ValueError("Invalid operation inventory pipeline list.")
    names = set()
    for pipeline in value["pipelines"]:
        if set(pipeline) != {"pipelineClass", "coverage", "reason", "equivalentTo", "upstreamTasks"}:
            raise ValueError("Invalid operation inventory pipeline record.")
        name = _identifier(pipeline["pipelineClass"])
        if name in names or pipeline["coverage"] not in {
            "executable",
            "equivalent",
            "contract-only",
            "research-blocked",
            "intentionally-excluded",
            "unreviewed",
        }:
            raise ValueError("Invalid operation inventory review decision.")
        names.add(name)
        if not isinstance(pipeline["reason"], str) or not 1 <= len(pipeline["reason"]) <= 2048:
            raise ValueError("Missing operation inventory review reason.")
        if not isinstance(pipeline["equivalentTo"], list) or len(pipeline["equivalentTo"]) > 32:
            raise ValueError("Invalid operation inventory equivalents.")
        for equivalent in pipeline["equivalentTo"]:
            _identifier(equivalent)
        tasks = pipeline["upstreamTasks"]
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= 512:
            raise ValueError("Missing operation inventory task surface.")
        seen_tasks = set()
        for task in tasks:
            if set(task) != {"task", "source", "workflowId"} or task["source"] not in {"auto", "modular", "export"}:
                raise ValueError("Invalid operation inventory task.")
            _identifier(task["task"])
            if task["workflowId"] is not None:
                _identifier(task["workflowId"])
            identity = (task["task"], task["source"], task["workflowId"])
            if identity in seen_tasks:
                raise ValueError("Duplicate operation inventory task.")
            seen_tasks.add(identity)
    return value
