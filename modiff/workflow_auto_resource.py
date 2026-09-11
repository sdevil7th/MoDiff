"""Resource planning for the existing concrete graph executor.

No model execution, installation, alternate graph format or source admission.
All model owners stay resident in the node cache, so budgets deliberately add
their complete reviewed envelopes instead of assuming undocumented eviction.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import time
from typing import Any, Callable

from modiff.auto_resource import READY_PROOF_STATUSES, build_auto_resource_plan, _hardware_snapshot
from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
from modiff.huggingface_cluster_admission import REVIEWED_CLUSTER_EXECUTION_CANDIDATES

_RESOURCE_LINK = re.compile(r"pipeline|component|state|model|encoder|unet|vae|loop_member", re.I)
_WORKLOAD = {
    "width": "width", "height": "height", "num_inference_steps": "steps", "steps": "steps",
    "guidance_scale": "guidanceScale", "num_frames": "numFrames", "batch_size": "batchSize",
    "num_images_per_prompt": "batchSize", "max_sequence_length": "maxSequenceLength",
    "duration": "duration", "audio_duration": "audioDuration", "strength": "strength",
}
_SETTINGS = {"dtype": "dtype", "device": "device", "offload_mode": "offloadMode", "auto_offload": "autoOffload", "quantization_mode": "quantizationMode"}
_DATA_MODULES = {"modules.Primitive", "modules.Text", "modules.Image", "modules.ImageOperations", "modules.Audio", "modules.Video"}
_DATA_ACTIONS = {("modules.Tensor", "SeededGenerator"), ("modules.Tensor", "AttentionArguments")}


def graph_material(graph: dict) -> dict:
    nodes = graph.get("nodes")
    paths = graph.get("paths")
    if not isinstance(nodes, dict) or not nodes or len(nodes) > 2048 or not isinstance(paths, list):
        raise ValueError("Auto needs a nonempty executable graph with at most 2048 nodes.")
    if len(paths) > 2048 or any(not isinstance(path, list) or len(path) > 2048 or any(not isinstance(item, str) for item in path) for path in paths):
        raise ValueError("The executable paths exceed the Auto planning limit.")
    order = list(dict.fromkeys(item for path in paths for item in path if isinstance(item, str)))
    if set(order) != set(nodes) or sum(map(len, paths)) > 32768:
        raise ValueError("Auto paths must cover every executable node exactly by identity.")
    for node_id, node in nodes.items():
        if not isinstance(node_id, str) or not isinstance(node, dict) or not isinstance(node.get("params"), dict):
            raise ValueError("Auto received an invalid executable node.")
        if not isinstance(node.get("module"), str) or not isinstance(node.get("action"), str):
            raise ValueError("Auto received an invalid executable node identity.")
        for param in node["params"].values():
            if not isinstance(param, dict):
                raise ValueError("Auto requires the existing API parameter contract.")
            if param.get("sourceId") is not None and param["sourceId"] not in nodes:
                raise ValueError(f"{node_id} refers to a missing input source.")
    # Validate dependency ordering independently of client paths. Repeated path
    # prefixes are normal exports; dependencies must still precede consumers.
    seen: set[str] = set()
    for node_id in order:
        dependencies = {p["sourceId"] for p in nodes[node_id]["params"].values() if p.get("sourceId")}
        if not dependencies.issubset(seen):
            raise ValueError(f"{node_id} has a cycle or an input ordered after its consumer.")
        seen.add(node_id)
    material = {"nodes": nodes, "paths": paths}
    if "loops" in graph:
        material["loops"] = graph["loops"]
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > 8 * 1024 * 1024:
        raise ValueError("The graph exceeds the Auto planning size limit.")
    return material


def workflow_graph_hash(graph: dict) -> str:
    return "sha256:workflow-auto-v1:" + hashlib.sha256(json.dumps(graph_material(graph), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


from modiff.workflow_auto_values import inspect_resource_value, DeferredResourceValue
from contextvars import ContextVar

_RESOLVED_FIELDS = ContextVar("workflow_auto_resolved_fields", default=None)

def _literal(nodes, node_id, field, visited=frozenset()):
    value = inspect_resource_value(nodes, node_id, field, visited)
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"{node_id}.{field} did not resolve to a finite data value.") from error
    if len(encoded.encode()) > 1_048_576:
        raise ValueError(f"{node_id}.{field} exceeds the resource control size limit.")
    captured = _RESOLVED_FIELDS.get()
    if captured is not None and field in nodes[node_id]["params"]:
        captured.setdefault(node_id, {})[field] = value
    return value


def _values(nodes: dict, node_id: str) -> dict:
    return {key: param.get("value") for key, param in nodes[node_id]["params"].items() if not param.get("sourceId")}


def _repository(value: Any) -> str:
    if isinstance(value, dict):
        return value.get("value", "") if value.get("source") == "hub" else ""
    return value if isinstance(value, str) else ""


def _consumers(nodes: dict, loader_id: str, loader_ids: set[str]) -> list[str]:
    found = {loader_id}
    while True:
        additions = {node_id for node_id, node in nodes.items() if node_id not in found | loader_ids and any(
            p.get("sourceId") in found and _RESOURCE_LINK.search(str(p.get("sourceKey", "")) + " " + key)
            for key, p in node["params"].items()
        )}
        additions.update(p["sourceId"] for node_id in found for key, p in nodes[node_id]["params"].items()
                         if p.get("sourceId") and "loop_member" in str(p.get("sourceKey", "")) and p["sourceId"] not in found)
        if not additions:
            return sorted(found - {loader_id})
        found.update(additions)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def _workload_values(nodes: dict, node_id: str):
    """Inspect the fields actually bound by the reviewed dispatch adapters.

    Keep the original API field name in resolvedFields so dispatch can compare
    connected values without rewriting the user's persisted graph. A carried
    loop value has no static upper bound; inspecting it must never execute the
    loop or substitute its initial value for all subsequent iterations.
    """
    node = nodes[node_id]
    reviewed_step = (node["module"], node["action"]) == (
        "modules.ModularDiffusers", "ReviewedModularWorkflowStep",
    )
    prefixes = ("", "state_input__", "iteration_input__") if reviewed_step else ("",)
    for name, target in _WORKLOAD.items():
        for prefix in prefixes:
            field = prefix + name
            if field in node["params"]:
                yield field, target, _literal(nodes, node_id, field)
    if reviewed_step and "iteration_bindings" in node["params"]:
        bindings = _literal(nodes, node_id, "iteration_bindings")
        if bindings is None:
            return
        if not isinstance(bindings, dict):
            raise ValueError(f"{node_id}.iteration_bindings must be an object.")
        for field, binding in bindings.items():
            if field not in _WORKLOAD:
                continue
            if not isinstance(binding, dict) or set(binding) != {"kind", "value"} or binding["kind"] != "constant":
                raise ValueError(
                    f"{node_id}.{field} changes inside an iteration and has no reviewed Auto resource bound. "
                    "Use a constant resource input or select Expert."
                )
            yield f"iteration_bindings.{field}", _WORKLOAD[field], binding["value"]


def _candidate_covers_workload(candidate: dict, form: dict) -> bool:
    generation = candidate.get("generation") or {}
    for key in set(_WORKLOAD.values()):
        if key not in form:
            continue
        actual = _number(generation.get(key))
        expected = _number(form[key])
        if key == "batchSize":
            # Historical recipes cover one item. Missing batch evidence is
            # never permission to reuse that envelope for a larger batch.
            if actual is None:
                if generation.get(key) is not None or expected != 1:
                    return False
            elif actual != expected:
                return False
        elif actual not in {None, expected}:
            return False
    return True


def _build_workflow_auto_plan(
    graph: dict, *, runtime_fingerprint: dict, local_models: list | None, data_dir: str,
    plan_recipe: Callable = build_auto_resource_plan, hardware: dict | None = None,
) -> dict:
    material = graph_material(graph)
    nodes = material["nodes"]
    graph_hash = workflow_graph_hash(graph)
    hardware = hardware if hardware is not None else _hardware_snapshot(runtime_fingerprint, data_dir)
    issues: list[str] = []
    loaders: dict[str, Any] = {}
    data_nodes: set[str] = set()
    preparation_nodes: set[str] = set()
    deferred_fields: list[dict] = []
    for node_id, node in nodes.items():
        if (node["module"], node["action"]) in _DATA_ACTIONS:
            data_nodes.add(node_id)
            continue
        values = _values(nodes, node_id)
        try:
            for field in ("model_type", "pipeline_class", "execution_profile_id", "repo_id", "model_id"):
                if node["params"].get(field, {}).get("sourceId"):
                    values[field] = _literal(nodes, node_id, field)
        except DeferredResourceValue as error:
            preparation_nodes.update(error.node_ids)
            deferred_fields.append(error.target)
            loaders[node_id] = None
            continue
        except (ValueError, TypeError) as error:
            issues.append(f"{node_id}: {error}")
            continue
        profiles, reason = resolve_execution_profiles_for_loader(node["module"], node["action"], values)
        if profiles and all(profile.execution_path.startswith("builtin-") for profile in profiles):
            data_nodes.add(node_id)
            continue  # Reviewed deterministic operations do not own model weights.
        if profiles:
            if reason or len(profiles) != 1:
                issues.append(f"{node_id}: the loader's exact execution profile is unresolved ({reason or 'ambiguous profile'}).")
            else:
                loaders[node_id] = profiles[0]
        elif re.search(r"load.*(?:model|pipeline|encoder)|modelsloader|automodelloader", node["action"], re.I):
            issues.append(f"{node_id}: no reviewed resource recipe exists for this model loader.")
        elif node["module"] not in _DATA_MODULES and not node["module"].startswith("modules.Diffusers") and node["module"] != "modules.ModularDiffusers":
            issues.append(f"{node_id}: {node['module']}.{node['action']} has no workflow Auto resource contract.")

    owned = set(loaders)
    for loader_id in loaders:
        owned.update(_consumers(nodes, loader_id, set(loaders)))
    for node_id, node in nodes.items():
        if node_id not in owned | data_nodes and node["module"] not in _DATA_MODULES and node["action"] != "Lora":
            issues.append(f"{node_id}: this resource operation has no connected, reviewed model owner.")
    planned: list[dict] = []
    recipe_cache: dict[str, dict] = {}
    patches: list[dict] = []
    total = {"systemRamBytes": 0, "vramBytes": 0, "diskFreeBytes": 0}
    for loader_id, profile in loaders.items():
        if profile is None:
            continue
        values = _values(nodes, loader_id)
        consumers = _consumers(nodes, loader_id, set(loaders))
        try:
            for field in ("repo_id", "model_id", "model_type", "pipeline_class", "execution_profile_id", "revision", *_SETTINGS):
                if field in nodes[loader_id]["params"]:
                    values[field] = _literal(nodes, loader_id, field)
            modes = {str(_literal(nodes, id, "mode")) for id in [loader_id, *consumers] if _literal(nodes, id, "mode")}
            if not modes:
                workflow = values.get("workflow_id")
                modes = {item["studioMode"] for item in REVIEWED_CLUSTER_EXECUTION_CANDIDATES if item["pipelineClass"] == profile.model_type and item["workflowId"] == workflow and item["studioMode"] in profile.modes}
                if modes == {"edit_image", "multi_image_reference_edit"}:
                    modes = {"multi_image_reference_edit"}
            mode = next(iter(modes)) if len(modes) == 1 else profile.modes[0] if len(profile.modes) == 1 else None
            if mode not in profile.modes:
                raise ValueError("The model's task is ambiguous; select an explicit mode on its loader or consumer.")
            repo = _repository(values.get("repo_id") or values.get("model_id"))
            if not repo:
                raise ValueError("Auto requires an exact installed Hub model repository.")
            form = {"modelType": profile.model_type, "modelRepo": repo, "mode": mode, "resourceMode": "auto",
                    "device": values.get("device", "cpu"), "dtype": values.get("dtype", "float32"),
                    "offloadMode": values.get("offload_mode", "none"), "autoOffload": values.get("auto_offload", False),
                    "quantizationMode": values.get("quantization_mode", "none")}
            device = str(form["device"])
            kind = hardware.get("accelerator", {}).get("kind")
            if device != "cpu" and kind and (device.split(":", 1)[0] != kind or (":" in device and device.rsplit(":", 1)[1] != "0")):
                raise ValueError("This device differs from the accelerator covered by the resource snapshot.")
            for id in [loader_id, *consumers]:
                for field, target, raw in _workload_values(nodes, id):
                    if raw is None:
                        continue
                    value = _number(raw)
                    if value is None:
                        raise ValueError(f"{id}.{field} is not a finite resource value.")
                    form[target] = max(form.get(target, 0), value)
            form_key = json.dumps(form, sort_keys=True)
            if form_key not in recipe_cache:
                recipe_cache[form_key] = plan_recipe({"form": form}, runtime_fingerprint=runtime_fingerprint, local_models=local_models, data_dir=data_dir)
            plan = recipe_cache[form_key]
            candidates = [candidate for candidate in plan.get("candidates", []) if isinstance(candidate, dict)
                          and candidate.get("canAutoRun") is True and candidate.get("proof", {}).get("status") in READY_PROOF_STATUSES
                          and candidate.get("modelRepo") == repo and candidate.get("modelType") == profile.model_type
                          and candidate.get("mode") == mode and candidate.get("loaderModule") == profile.loader_module
                          and candidate.get("loaderAction") == profile.loader_action and candidate.get("executionPath") == profile.execution_path
                          and candidate.get("dtype") == form["dtype"] and candidate.get("quantizationMode", "none") == form["quantizationMode"]]
            revision = values.get("revision")
            if revision:
                candidates = [candidate for candidate in candidates if (candidate.get("artifactResolution", {}).get("resolved", {}).get("revision") or candidate.get("artifactRevision")) == revision]
            candidates = [candidate for candidate in candidates if _candidate_covers_workload(candidate, form)]
            candidates.sort(key=lambda candidate: candidate.get("offloadMode") != form["offloadMode"])
            if not candidates:
                batch_reason = (
                    f"No accepted recipe covers batchSize={form['batchSize']:g}. "
                    "Use a covered batch size or select Expert."
                    if form.get("batchSize", 1) > 1 else None
                )
                raise ValueError(plan.get("blockingReason") or batch_reason or "No accepted recipe preserves this graph's model, precision and requested workload.")
            candidate = candidates[0]
            envelope = candidate.get("requirements", {})
            requirements = envelope
            if isinstance(envelope, dict) and "minimum" in envelope:
                by_offload = envelope.get("offloadRequirements") or {}
                requirements = by_offload.get(candidate.get("offloadMode")) or (envelope.get("fullResidency") if candidate.get("offloadMode") == "none" else None) or envelope["minimum"]
            if not isinstance(requirements, dict) or _number(requirements.get("systemRamBytes")) is None or _number(requirements.get("vramBytes")) is None:
                raise ValueError("The accepted recipe has no complete combined-memory envelope.")
            for key in total:
                total[key] += int(_number(requirements.get(key)) or 0)
            settings = {"offloadMode": candidate.get("offloadMode", "none"), "autoOffload": candidate.get("autoOffload", False)}
            for field, key in _SETTINGS.items():
                if key in settings and field in nodes[loader_id]["params"] and values.get(field) != settings[key]:
                    if nodes[loader_id]["params"][field].get("sourceId"):
                        raise ValueError(f"Connected {field} must select the planned value {settings[key]}.")
                    patches.append({"nodeId": loader_id, "field": field, "value": settings[key]})
            planned.append({"nodeId": loader_id, "modelType": profile.model_type, "mode": mode, "repository": repo,
                            "candidateId": candidate["id"], "proofStatus": candidate["proof"]["status"],
                            "consumers": consumers, "requirements": requirements, "settings": {**form, **settings}})
        except DeferredResourceValue as error:
            preparation_nodes.update(error.node_ids)
            deferred_fields.append(error.target)
        except (ValueError, TypeError) as error:
            issues.append(f"{loader_id}: {error}")
    adapters = []
    for node_id, node in nodes.items():
        if re.search("lora|adapter|quantconfig", node["action"], re.I) and node_id not in loaders:
            try:
                from modiff.auxiliary_lora import lora_resource_requirement
                requirements = lora_resource_requirement(node_id, node)
                # One descriptor may load into several independent model owners.
                owners = {target for target, loader in nodes.items() if target in loaders and any(p.get("sourceId") == node_id for p in loader["params"].values())}
                if not owners:
                    raise ValueError("The adapter's model owners cannot be resolved from its direct loader connections.")
                for key in ("systemRamBytes", "vramBytes"):
                    total[key] += requirements[key] * len(owners)
                adapters.append({"nodeId": node_id, "loaderIds": sorted(owners), **requirements})
            except (ValueError, TypeError, OSError) as error:
                issues.append(f"{node_id}: {error}")
    accelerator = hardware.get("accelerator", {})
    system = hardware.get("systemMemory", {})
    disk = hardware.get("offloadDisk", {})
    available = {"systemRamBytes": system.get("availableBytes"), "vramBytes": accelerator.get("freeBytes"), "diskFreeBytes": disk.get("freeBytes")}
    # Shared accelerator bytes come from system RAM, never an additional pool.
    topology = str(accelerator.get("memoryKind", ""))
    shared = topology in {"shared", "unified", "shared_system", "system_shared"} or accelerator.get("sharedMemory") is True
    if shared:
        # The standard planner intentionally publishes dedicated VRAM in
        # freeBytes. This combined envelope also needs the actual shared pool,
        # which is already present in the same runtime hardware snapshot.
        devices = hardware.get("runtime", {}).get("hardware", {}).get("devices", [])
        device = next((item for item in devices if item.get("memory_kind") in {"shared", "unified"} and item.get("type") == accelerator.get("kind")), {})
        shared_free = _number(device.get("shared_memory_free"))
        accessible_free = _number(device.get("torch_vram_free"))
        if shared_free is not None or accessible_free is not None:
            available["vramBytes"] = min(value for value in (shared_free, accessible_free, _number(system.get("availableBytes"))) if value is not None)
    retained_total = dict(total)
    from modiff.workflow_auto_lifecycle import plan_owner_lifetimes
    schedule = plan_owner_lifetimes(material, planned, adapters)
    # Independent owners use their lower live envelope even when the retained
    # estimate happens to fit. Dispatch-time recipe/history selection must not
    # silently discard a release plan and keep two large pipelines resident.
    use_schedule = len(planned) > 1 and bool(schedule["releases"]) and any(
        schedule["peak"][key] < total[key] for key in ("systemRamBytes", "vramBytes")
    )
    if use_schedule:
        total = {**total, **schedule["peak"]}
    memory_required = (schedule["sharedPeakBytes"] if shared and use_schedule else total["systemRamBytes"] + (total["vramBytes"] if shared else 0))
    for key, required in {**total, "systemRamBytes": memory_required}.items():
        free = _number(available[key])
        if required and (free is None or required > free):
            issues.append(f"Combined {key}: needs {required} bytes; {int(free) if free is not None else 'unknown'} bytes available.")
    # The hash must describe exactly the patches returned to the client. Any
    # deferred supplier postpones all mutations until dispatch-time replanning.
    patches = patches if not issues and not preparation_nodes else []
    patched_graph = deepcopy(material)
    for patch in patches:
        patched_graph["nodes"][patch["nodeId"]]["params"][patch["field"]]["value"] = patch["value"]
    return {"schemaVersion": 1, "graphHash": graph_hash, "plannedGraphHash": workflow_graph_hash(patched_graph),
            "canAutoRun": not issues, "issues": issues, "loaders": planned, "adapters": adapters, "patches": patches,
            "requirements": total, "retainedRequirements": retained_total, "available": available, "sharedMemory": shared,
            "schedule": schedule if use_schedule else None,
            "requiresPreparation": bool(preparation_nodes), "preparationNodeIds": sorted(preparation_nodes), "deferredFields": deferred_fields,
            "strategy": "dependency_order_release_owners" if use_schedule else "dependency_order_retained_owners", "checkedAt": int(time.time() * 1000),
            "message": ("Run will first execute the data suppliers and validate their actual resource values before loading models. " if preparation_nodes else "") + ("Auto finishes dependent work and releases completed model owners before loading the next owner. Downstream material outputs are retained." if use_schedule else "Auto uses the existing dependency order and budgets all retained model owners.") + " Shared loader nodes are counted once. Model, precision and creative controls are preserved."}


def build_workflow_auto_plan(graph, **kwargs):
    captured = {}
    token = _RESOLVED_FIELDS.set(captured)
    try:
        result = _build_workflow_auto_plan(graph, **kwargs)
        result["resolvedFields"] = captured
        return result
    finally:
        _RESOLVED_FIELDS.reset(token)
