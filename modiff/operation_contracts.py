"""Read-only operation identities projected from existing Modular node configs.

These declarations are neither graph recipes nor execution/connection authority.
In particular, equal tensor types or semantic names do not establish compatible
conditioning across pipelines. Keep the enclosing pipelineClass with each port.
This module deliberately imports no model libraries.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import re

OPERATION_CONTRACT_SCHEMA_VERSION = 1


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
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_RESERVED = frozenset({"__proto__", "prototype", "constructor"})


def _identifier(value):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or value in _RESERVED:
        raise ValueError("Invalid operation-contract identifier.")
    return value


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
                    raw_types = param.to_dict().get("type")
                    types = [raw_types] if isinstance(raw_types, str) else raw_types
                    if not isinstance(types, (list, tuple)) or not 1 <= len(types) <= 16:
                        raise ValueError("Invalid operation-contract port types.")
                    types = sorted({_identifier(value) for value in types})
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
