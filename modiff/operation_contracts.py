"""Read-only operation identities projected from existing node/adapter configs.

These declarations are neither graph recipes nor execution/connection authority.
In particular, equal tensor types or semantic names do not establish compatible
conditioning across pipelines. Keep the enclosing pipelineClass with each port.
This module deliberately imports no model libraries.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import re

OPERATION_CONTRACT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ModularStageOperation:
    id: str
    action: str
    label: str


# Names of existing actions, not a second implementation or model registry.
MODULAR_STAGE_OPERATIONS = {
    "text_encoder": ModularStageOperation("diffusion.encode_prompt", "EncodePrompt", "Encode Prompt"),
    "image_encoder": ModularStageOperation("diffusion.image_embeddings", "ImageEmbeddings", "Image Embeddings"),
    "vae_encoder": ModularStageOperation("diffusion.encode_image", "ImageEncode", "Encode Image"),
    "denoise": ModularStageOperation("diffusion.denoise", "Denoise", "Denoise"),
    "decoder": ModularStageOperation("diffusion.decode_latents", "DecodeLatents", "Decode Latents"),
    "controlnet": ModularStageOperation("diffusion.controlnet", "Controlnet", "ControlNet"),
    "ip_adapter": ModularStageOperation("diffusion.ip_adapter", "IPAdapter", "IP-Adapter Embeddings"),
}

# Upstream stage names, shared across families. Additional media processing
# stages retain their own identities instead of being forced into denoising.
WORKFLOW_STAGE_OPERATIONS = {
    **{key: value.id for key, value in MODULAR_STAGE_OPERATIONS.items()},
    "decode": "diffusion.decode_latents",
    "video_encoder": "diffusion.encode_video",
    "semantic_generator": "diffusion.generate_semantics",
    "prompt_upsample": "diffusion.rewrite_prompt",
    "before_encode": "diffusion.prepare_media",
    "after_decode": "diffusion.postprocess_media",
    "duration": "diffusion.prepare_duration",
    "condition_encoder": "diffusion.encode_condition",
    "reference_encoder": "diffusion.encode_reference",
}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_RESERVED = frozenset({"__proto__", "prototype", "constructor"})


def _identifier(value):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or value in _RESERVED:
        raise ValueError("Invalid operation-contract identifier.")
    return value


def _port_types(raw_types):
    types = [raw_types] if isinstance(raw_types, str) else raw_types
    if not isinstance(types, (list, tuple)) or not 1 <= len(types) <= 16:
        raise ValueError("Invalid operation-contract port types.")
    return sorted({_identifier(value) for value in types})


def with_operation_semantics(contract, *, workflow_id=None, values=None):
    """Add scoped semantic and binding metadata without claiming type equivalence."""
    from copy import deepcopy

    result = deepcopy(contract)
    result["workflowId"] = workflow_id
    result["binding"] = {"values": dict(values or {})}
    for port in result["ports"]:
        types = set(port["types"])
        name = port["semanticName"]
        kind = "value"
        if "pipeline" in port["roles"]:
            kind = "pipeline"
        elif "component" in port["roles"]:
            kind = "component"
        elif types & {"modular_workflow_state", "modular_route_state"}:
            kind = "state"
        elif types & {"embeddings", "image_embeddings", "conditioning", "controlnet_bundle", "ip_adapter_bundle"}:
            kind = "conditioning"
        elif "latent" in name and types & {"tensor", "latents"}:
            kind = "latents"
        elif types & {"tensor", "object", "dict", "list", "custom_lora", "quant_config"}:
            kind = "opaque"
        elif types & {"image", "video", "audio", "prediction_map"}:
            kind = "media"
        elif not types <= {"string", "str", "text", "int", "float", "number", "bool", "boolean", "seed"}:
            kind = "opaque"
        scoped = kind in {"component", "conditioning", "latents", "state", "pipeline", "opaque"}
        port["semantics"] = {
            "kind": kind,
            "scope": (f"{result['pipelineClass']}:{workflow_id}" if kind == "state" and workflow_id
                      else result["pipelineClass"]) if scoped else None,
            "state": None,
            "owner": "same_loader" if kind in {"component", "conditioning", "latents", "state", "pipeline"} else "none",
            "members": [],
        }
    return result


def build_pipeline_operation_contract(
    modules: Mapping, *, pipeline_class: str, task: str, operation_id: str,
    node_key: str, field_overrides: Mapping | None = None, loader: bool = False,
) -> dict | None:
    """Describe an existing whole-pipeline action using its owner's field overlay.

    Hidden fields remain declared: visibility is presentation, not readiness or
    permission. Options, defaults, conditional requirements and tensor semantics
    still belong to the existing dynamic schema and execution preflight.
    """
    _identifier(pipeline_class)
    _identifier(task)
    if len(operation_id.split(".")) != 2 or len(node_key.split(".")) != 3 or not node_key.startswith("modules."):
        raise ValueError("Invalid operation-contract action.")
    for part in (*operation_id.split("."), *node_key.split(".")):
        _identifier(part)
    module, action = node_key.rsplit(".", 1)
    declaration = modules.get(module, {}).get(action)
    if declaration is None:
        return None
    ports = []
    for name, base in declaration["params"].items():
        field = {**base, **(field_overrides or {}).get(name, {})}
        if "type" not in field or field.get("display") == "button":
            continue
        direction = "output" if field.get("display") == "output" else "input"
        ports.append({
            "name": _identifier(name),
            "semanticName": name,
            "direction": direction,
            "roles": ["pipeline" if name == "pipeline" else "value"],
            "types": _port_types(field["type"]),
            "required": direction == "input" and field.get("required") is True,
            "hidden": field.get("hidden") is True,
        })
    if len(ports) > 128:
        raise ValueError("Too many operation-contract ports.")
    decomposition = "loader" if loader else "pipeline"
    return {
        "pipelineClass": pipeline_class, "task": task, "operationId": operation_id,
        "nodeKey": node_key, "nodeType": decomposition, "blockName": None,
        "decomposition": decomposition, "support": "declared", "ports": ports,
    }


def build_modular_operation_contracts(configs: Mapping, modules: Mapping) -> list[dict]:
    """Project declarations without constructing pipelines or resolving blocks.

    ``configs`` comes from ModiffPipelineRegistry. Deserialized custom previews
    have no reviewed node_specs and are handled by their existing trust boundary.
    A missing stage/action is omitted, never inferred from a pipeline class name.
    """
    if len(configs) > 512:
        raise ValueError("Too many operation-contract pipelines.")
    actions = modules.get("modules.ModularDiffusers", {})
    result = []
    for pipeline_class, config in sorted(configs.items()):
        _identifier(pipeline_class)
        if config.node_specs is None:
            continue
        node_params = config.node_params
        for node_type, operation in MODULAR_STAGE_OPERATIONS.items():
            spec = config.node_specs.get(node_type)
            if spec is None or operation.action not in actions:
                continue
            formatted = node_params[node_type]
            ports = []
            ports_by_key = {}
            for group, names_key, direction, kind, required_key in (
                ("inputs", "input_names", "input", "value", "required_inputs"),
                ("model_inputs", "model_input_names", "input", "component", "required_model_inputs"),
                ("outputs", "output_names", "output", "value", None),
            ):
                params = spec.get(group, [])
                names = formatted[names_key]
                required = spec.get(required_key, []) if required_key else []
                if len(params) != len(names) or not set(required).issubset({p.name for p in params}):
                    raise ValueError("Invalid operation-contract port declaration.")
                for param, name in zip(params, names, strict=True):
                    _identifier(name)
                    _identifier(param.name)
                    types = _port_types(param.to_dict().get("type"))
                    key = (direction, name)
                    if key in ports_by_key:
                        existing = ports_by_key[key]
                        # Some conditioning bundles supply both pipeline values
                        # and components through one existing socket.
                        if (
                            kind in existing["roles"]
                            or existing["semanticName"] != param.name
                            or existing["types"] != types
                        ):
                            raise ValueError("Conflicting operation-contract port declaration.")
                        existing["roles"].append(kind)
                        existing["required"] = existing["required"] or param.name in required
                        continue
                    port = {
                        "name": name,
                        "semanticName": param.name,
                        "direction": direction,
                        "roles": [kind],
                        "types": types,
                        "required": param.name in required,
                        "hidden": param.to_dict().get("hidden") is True,
                    }
                    ports_by_key[key] = port
                    ports.append(port)
            if len(ports) > 128:
                raise ValueError("Too many operation-contract ports.")
            block_name = spec.get("block_name") or None
            if block_name is not None:
                _identifier(block_name)
            result.append(
                {
                    "pipelineClass": pipeline_class,
                    "task": None,
                    "operationId": operation.id,
                    "nodeKey": f"modules.ModularDiffusers.{operation.action}",
                    "nodeType": node_type,
                    "blockName": block_name,
                    "decomposition": "block" if block_name else "bundle",
                    "support": "declared",
                    "ports": ports,
                }
            )
    return result
