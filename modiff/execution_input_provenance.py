"""Bounded evidence of resolved node call arguments, not inferred model defaults.

Only explicitly named generation fields enter local output history. Never walk
arbitrary objects (model components, tensors, media or credentials). An output's
receipt contains only records in its actual submitted graph ancestry.
"""

import math
import json
from copy import deepcopy


FIELD_NAMES = {
    "prompt": "prompt", "prompt_2": "prompt2", "prompt_3": "prompt3",
    "negative_prompt": "negativePrompt", "negative_prompt_2": "negativePrompt2",
    "seed": "seed", "width": "width", "height": "height",
    "num_inference_steps": "steps", "steps": "steps",
    "guidance_scale": "guidanceScale", "true_cfg_scale": "trueCfgScale",
    "max_sequence_length": "maxSequenceLength", "num_images_per_prompt": "imagesPerPrompt",
    "num_frames": "numFrames", "fps": "fps", "frame_rate": "fps",
    "strength": "strength", "repo_id": "repo", "model_id": "repo",
    "revision": "revision", "dtype": "dtype", "device": "device",
    "model_type": "modelType", "pipeline_class": "modelType",
    "auto_offload": "autoOffload", "offload_mode": "offloadMode",
    "output_type": "outputType", "quant_config": "quantConfig",
    "conditioning_scale": "conditioningScale", "control_mode": "controlMode",
    "control_guidance_start": "controlGuidanceStart", "control_guidance_end": "controlGuidanceEnd",
    "prompt_embeds_scale": "reduxPromptEmbedsScale",
    "pooled_prompt_embeds_scale": "reduxPooledPromptEmbedsScale",
}
NODE_FIELD_NAMES = {
    ('DiffusersImage', 'ControlComponent'): {
        'model_id': 'controlComponentRepo', 'revision': 'controlComponentRevision',
        'shared_conditions': 'controlComponentSharedConditions',
    },
    ('DiffusersImage', 'ImagePromptAdapter'): {
        'adapter_model': 'ipAdapterRepo', 'revision': 'ipAdapterRevision',
        'weight_name': 'ipAdapterWeight', 'expected_sha256': 'ipAdapterSha256',
        'image_encoder_model': 'ipAdapterEncoderRepo',
        'image_encoder_revision': 'ipAdapterEncoderRevision',
        'image_encoder_sha256': 'ipAdapterEncoderSha256',
        'scale': 'ipAdapterScale', 'layer_scales': 'ipAdapterLayerScales',
    },
}


def _field_names(node):
    module = str(node.get('module', '')).removeprefix('modules.')
    # The JSON/data processor is an executed operation, not another model.
    # Keep its exact identity in the receipt without making the actual image
    # loader ambiguous. Other model-backed text processors retain modelType.
    if ((module, node.get('action')) == ('Text', 'ProcessText')
            and node.get('fields', {}).get('pipeline_class', {}).get('value') == 'BuiltinDataOperationV1'):
        return {**FIELD_NAMES, 'pipeline_class': 'dataOperationType'}
    return NODE_FIELD_NAMES.get((module, node.get('action')), FIELD_NAMES)

MAX_STRING = 16_384
MAX_ITEMS = 16
MAX_NODES = 256
MAX_RECORD_BYTES = 65_536
MAX_RECEIPT_BYTES = 262_144
# These fields already have top-level StudioOutput display slots. Other captured
# values remain available in the receipt without inventing new form semantics.
OUTPUT_FIELDS = {"prompt", "negativePrompt", "seed", "width", "height", "steps", "guidanceScale", "repo", "modelType"}


def _bounded_value(value):
    if value is None or type(value) is bool:
        return value, None
    if type(value) in (int, float):
        if abs(value) <= 2**53 - 1 and math.isfinite(value):
            # JSON/JavaScript has one numeric domain; 3 and 3.0 are compatible
            # evidence, while the boolean path above remains distinct.
            return int(value) if int(value) == value else value, None
        return None, "non-finite-or-unsafe-number"
    if type(value) is str:
        return (value, None) if len(value) <= MAX_STRING else (None, "oversize-string")
    if type(value) in (tuple, list):
        if len(value) > MAX_ITEMS:
            return None, "oversize-list"
        values = []
        for item in value:
            if type(item) in (tuple, list, dict):
                return None, "unsupported-value"
            safe, reason = _bounded_value(item)
            if reason:
                return None, reason
            values.append(safe)
        return values, None
    return None, "unsupported-value"


def capture_generation_inputs(node_id, node, args, overrides=None):
    """Snapshot allowlisted resolved arguments immediately before dispatch."""
    fields, omitted = {}, {}
    remaining_bytes = MAX_RECORD_BYTES
    params = node.get("params") or {}
    field_names = _field_names(node)
    for key, value in args.items():
        if key not in field_names:
            continue
        # Model selectors use a small repository selector envelope. Deliberately
        # do not recurse into arbitrary dictionaries or stringify runtime objects.
        if key in {"repo_id", "model_id", "adapter_model", "image_encoder_model"} and type(value) is dict:
            value = value.get("value") if value.get("source") in {"hf", "hub", "local"} else value
        safe, reason = _bounded_value(value)
        if not reason:
            size = len(json.dumps(safe, ensure_ascii=True).encode("ascii"))
            if size > remaining_bytes:
                reason = "record-size-limit"
            else:
                remaining_bytes -= size
        if reason:
            omitted[key] = reason
            continue
        param = params.get(key) or {}
        origin = {"source": "literal"}
        if overrides and key in overrides:
            origin = {"source": "override"}
        elif param.get("sourceId") and param.get("sourceKey"):
            origin = {
                "source": "connected", "sourceNodeId": str(param["sourceId"]),
                "sourcePortId": str(param["sourceKey"]),
            }
        fields[key] = {"value": safe, **origin}
    return {
        "nodeId": str(node_id), "module": node["module"], "action": node["action"],
        "fields": fields, "omittedFields": omitted,
    }


def bind_generation_input_origins(node_id, node, record, overrides=None, *, source_fields=None):
    """Attach this graph invocation's wiring to the successful call snapshot.

    NodeBase snapshots after normalization/post-processing. Exact adapters may
    replace that snapshot at their own consumed-input boundary. Cache hits keep
    the snapshot of the call that produced the reused output, not raw form text.
    """
    if not isinstance(record, dict):
        return None
    result = deepcopy(record)
    result.update(nodeId=str(node_id), module=node["module"], action=node["action"])
    params = node.get("params") or {}
    for key, field in result["fields"].items():
        # Backend adapters can rename a consumed argument (for example a
        # generic guidance control to true_cfg_scale). Keep its original wire.
        source = (source_fields or {}).get(key, key)
        param = params.get(source) or {}
        origin = {"source": "literal"}
        if overrides and source in overrides:
            origin = {"source": "override"}
        elif param.get("sourceId") and param.get("sourceKey"):
            origin = {"source": "connected", "sourceNodeId": str(param["sourceId"]),
                      "sourcePortId": str(param["sourceKey"])}
        result["fields"][key] = {"value": field["value"], **origin}
    return result


def build_resolved_execution_inputs(graph, records, *, task_id, attempt_index, node_id):
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, dict) or node_id not in nodes:
        return None
    ancestors, pending = set(), [node_id]
    while pending:
        current = pending.pop()
        if current in ancestors or current not in nodes:
            continue
        ancestors.add(current)
        for param in (nodes[current].get("params") or {}).values():
            if isinstance(param, dict) and param.get("sourceId") and param.get("sourceKey"):
                pending.append(str(param["sourceId"]))
    captured = [deepcopy(records[key]) for key in sorted(ancestors) if key in records]
    if not captured:
        return None
    incomplete = len(captured) > MAX_NODES
    bounded = []
    remaining_bytes = MAX_RECEIPT_BYTES
    for record in captured[:MAX_NODES]:
        size = len(json.dumps(record, ensure_ascii=True).encode("ascii"))
        if size > remaining_bytes:
            incomplete = True
            break
        bounded.append(record)
        remaining_bytes -= size
    captured = bounded
    candidates, unavailable = {}, set()
    for record in captured:
        field_names = _field_names(record)
        for key, field in record["fields"].items():
            canonical = field_names[key]
            values = candidates.setdefault(canonical, [])
            if not any(type(existing) is type(field["value"]) and existing == field["value"] for existing in values):
                values.append(field["value"])
        unavailable.update(field_names[key] for key in record["omittedFields"])
    # Missing records can be unexecuted branches. Never manufacture their values.
    missing = sorted(ancestors - records.keys())
    summary = {key: values[0] for key, values in candidates.items() if len(values) == 1 and key not in unavailable}
    ambiguous = sorted(key for key, values in candidates.items() if len(values) > 1)
    if incomplete:
        unavailable.update(candidates)
        summary = {}
    return {
        "schemaVersion": 1, "source": "backend-execution", "taskId": str(task_id),
        "attemptIndex": int(attempt_index or 0), "nodeId": str(node_id),
        "nodes": captured, "summary": summary, "ambiguousFields": ambiguous,
        "unavailableFields": sorted(unavailable), "uncapturedNodeIds": missing,
        "truncated": incomplete,
    }


def apply_resolved_execution_inputs(output, receipt):
    """A trusted, identity-matching receipt takes precedence over form fallbacks."""
    output.pop("resolvedExecutionInputs", None)
    if not isinstance(receipt, dict) or (
        receipt.get("schemaVersion") != 1 or receipt.get("source") != "backend-execution"
        or receipt.get("taskId") != str(output.get("taskId"))
        or receipt.get("attemptIndex") != int(output.get("attemptIndex") or 0)
        or receipt.get("nodeId") != str(output.get("nodeId"))
    ):
        return output
    output["resolvedExecutionInputs"] = deepcopy(receipt)
    summary = receipt["summary"]
    for key in OUTPUT_FIELDS:
        if receipt["truncated"] or key in receipt["ambiguousFields"] or key in receipt["unavailableFields"]:
            output.pop(key, None)
        elif key in summary:
            value = summary[key]
            # Arrays of prompts belong in per-node evidence, not in a single
            # string field that would silently present a different prompt.
            if value is None or isinstance(value, list):
                output.pop(key, None)
            else:
                output[key] = value
    # Old hashes/compatibility receipts describe the form, not resolved wires.
    if receipt["nodes"]:
        output.pop("promptSettingsHash", None)
        output["exactTemplateCompatible"] = False
        for key in ("provenance", "backendProvenance"):
            if isinstance(output.get(key), dict):
                output[key] = {**output[key], "exactTemplateCompatible": False}
                output[key].pop("promptSettingsHash", None)
    return output
