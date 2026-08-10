# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
from collections.abc import Mapping
from copy import deepcopy

from .pipeline_schema import PROTOTYPE_SENSITIVE_FIELD_NAMES
from .pipeline_schema import MoDiffPipelineConfig as PipelineConfig

from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    offload_mode_param,
)
from modiff.model_artifact_catalog import resolve_model_revision
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST

from . import components
from .modular_utils import require_immutable_hub_revision


_DECLARATIVE_SIDECAR_ACTIONS = {"show", "hide", "value", "signal"}
_EXECUTION_UNSUPPORTED_MESSAGE = (
    "Dynamic Block Node is contract-preview only in this release. Upstream Modular configs can direct imports of "
    "installed libraries even when trust_remote_code is disabled, so execution remains unavailable until MoDiff "
    "has a reviewed component-library allowlist. Use the built-in generic Modular Diffusers nodes to run models."
)


def _require_json_boolean_trust(value):
    if type(value) is not bool:
        raise TypeError("Dynamic Block trust_remote_code must be a JSON boolean.")
    return value


def _validate_sidecar_field_action(value, *, field_name, event_name, field_definitions):
    """Reject code callbacks and mutations outside the published field contract."""

    if value is None:
        return
    if isinstance(value, str):
        raise ValueError(
            f"Dynamic Block sidecar field {field_name!r} must not define server-executing {event_name} metadata."
        )
    if isinstance(value, list):
        for item in value:
            _validate_sidecar_field_action(
                item,
                field_name=field_name,
                event_name=event_name,
                field_definitions=field_definitions,
            )
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"Dynamic Block sidecar field {field_name!r} has malformed {event_name} metadata.")

    action = value.get("action")
    if action in {"exec", "create"}:
        raise ValueError(f"Dynamic Block sidecar field {field_name!r} must not define {event_name} action {action!r}.")
    if action is not None and action not in _DECLARATIVE_SIDECAR_ACTIONS:
        raise ValueError(f"Dynamic Block sidecar field {field_name!r} has unsupported {event_name} action {action!r}.")

    allowed_fields = set(field_definitions)

    def validate_visibility_map(mapping):
        if not isinstance(mapping, Mapping):
            raise ValueError(
                f"Dynamic Block sidecar field {field_name!r} has malformed {event_name} visibility data."
            )
        for targets in mapping.values():
            target_names = targets if isinstance(targets, list) else [targets]
            if any(not isinstance(target, str) or target not in allowed_fields for target in target_names):
                raise ValueError(
                    f"Dynamic Block sidecar field {field_name!r} targets an unknown contract field."
                )

    if action is None:
        validate_visibility_map(value)
    elif action in {"show", "hide"}:
        validate_visibility_map(value.get("data", {}))
    else:
        target = value.get("target")
        if not isinstance(target, str) or target not in allowed_fields:
            raise ValueError(f"Dynamic Block sidecar field {field_name!r} targets an unknown contract field.")
        if action == "value":
            prop = value.get("prop", "value")
            if prop not in {"value", "hidden", "disabled", "options", "fieldOptions", "display"}:
                raise ValueError(
                    f"Dynamic Block sidecar field {field_name!r} defines unsupported value property {prop!r}."
                )
        else:
            target_definition = field_definitions.get(target)
            target_display = target_definition.get("display") if isinstance(target_definition, Mapping) else None
            if target_display not in {"input", "output"}:
                raise ValueError(
                    f"Dynamic Block sidecar field {field_name!r} signal target must be an input or output field."
                )


def _custom_node_contract(custom_config):
    """Return an isolated, declarative-only DynamicBlock node contract."""

    node_params = custom_config.node_params
    if not isinstance(node_params, Mapping):
        raise ValueError("Dynamic Block sidecar node_params must be a JSON object.")
    raw_contract = node_params.get("custom")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("Dynamic Block sidecar must define a 'custom' node contract object.")

    contract = deepcopy(dict(raw_contract))
    raw_params = contract.get("params")
    if not isinstance(raw_params, Mapping):
        raise ValueError("Dynamic Block sidecar custom.params must be a JSON object.")
    field_definitions = dict(raw_params)

    sanitized_params = {}
    for field_name, raw_field in raw_params.items():
        if (
            not isinstance(field_name, str)
            or field_name in PROTOTYPE_SENSITIVE_FIELD_NAMES
            or not isinstance(raw_field, Mapping)
        ):
            raise ValueError("Dynamic Block sidecar fields must map string names to JSON objects.")
        field = deepcopy(dict(raw_field))
        for event_name in ("onChange", "onSignal"):
            if event_name in field:
                _validate_sidecar_field_action(
                    field[event_name],
                    field_name=field_name,
                    event_name=event_name,
                    field_definitions=field_definitions,
                )
        sanitized_params[field_name] = field
    contract["params"] = sanitized_params

    for names_key in ("model_input_names", "input_names", "output_names"):
        names = contract.get(names_key, [])
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError(f"Dynamic Block sidecar custom.{names_key} must be a JSON string array.")
        contract[names_key] = list(names)

    for text_key in ("label", "color"):
        value = contract.get(text_key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Dynamic Block sidecar custom.{text_key} must be a string when provided.")
    return contract


def _server():
    from modiff.server import server

    return server


class DynamicBlockNode(NodeBase):
    label = "Dynamic Block Node"
    resizable = True
    style = {"minWidth": 300}
    skipParamsCheck = True
    node_type = "custom"

    def send_node_definition_with_meta(self, params, label=None, header_color=None):
        """Extended send_node_definition that also updates node label and header color.

        Args:
            params: dict of node params (same as send_node_definition)
            label: optional string to rename the node title in the UI
            header_color: optional CSS color string for the node header bar,
                          e.g. "#e67e22", "rgb(100,150,200)", "orange"
        """
        if not self._sid or not self.node_id:
            return

        current_server = _server()
        describe = getattr(current_server, "describe_node_params", None)
        message = {
            "type": "node_definition",
            "node": self.node_id,
            "params": describe(params) if callable(describe) else params,
        }
        if label:
            message["label"] = label
        if header_color:
            message["style"] = {"headerColor": header_color}
        current_server.queue_message(message, self._sid)

    params = {
        "repo_id": {
            "label": "Custom Block",
            "display": "autocomplete",
            "type": "string",
            "default": "",
            "value": "",
            "options": {
                "": "",
                "diffusers/FLUX.2-klein-4B-modular": "FLUX.2-klein-4B",
            },
            "fieldOptions": {"noValidation": True},
        },
        "load_block_button": {
            "label": "Preview Custom Block Contract",
            "display": "ui_button",
            "value": False,
            "onChange": "update_node",
        },
        "device": {"label": "Device", "type": "string", "value": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "auto_offload": {"label": "Enable Auto Offload", "type": "boolean", "value": False},
        "offload_mode": offload_mode_param(
            modes=[OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK]
        ),
        "trust_remote_code": {
            "label": "Trust Remote Code",
            "type": "boolean",
            "value": False,
            "description": "Execution is disabled on this contract-preview-only legacy node.",
        },
        "revision": {
            "label": "Revision",
            "type": "string",
            "value": "",
            "description": "Required immutable 40-character Hugging Face commit hash.",
        },
        "doc": {
            "label": "Doc",
            "type": "string",
            "display": "output",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_input_names = []

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
        super().__del__()

    def _get_verified_custom_config(self, repo_id, revision=None):
        revision = resolve_model_revision(repo_id, revision)
        revision = require_immutable_hub_revision(repo_id, revision, required=True)
        return PipelineConfig.load_verified(
            repo_id,
            source="hub",
            revision=revision,
        )

    def _get_custom_config(self, repo_id, revision=None):
        """Compatibility wrapper returning the verified sidecar configuration."""

        return self._get_verified_custom_config(repo_id, revision).config

    def update_node(self, values, ref):
        if not values.get("repo_id", ""):
            self.send_node_definition({})
            return

        trust_remote_code = _require_json_boolean_trust(values.get("trust_remote_code", False))
        if trust_remote_code:
            raise ValueError(
                "Dynamic Block contract preview requires Trust Remote Code off; repository code is not "
                "authorized by this legacy node."
            )
        repo_id = values.get("repo_id", "")
        revision = values.get("revision")
        verified_config = self._get_verified_custom_config(repo_id, revision)
        node_config = _custom_node_contract(verified_config.config)

        custom_params = node_config["params"]
        self._model_input_names = node_config.get("model_input_names", [])

        node_label = node_config.get("label")
        node_color = node_config.get("color")

        if not node_label:
            node_label = repo_id.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ").title()

        self.send_node_definition_with_meta(
            custom_params,
            label=node_label,
            header_color=node_color,
        )

    def execute(
        self,
        repo_id,
        device,
        auto_offload,
        trust_remote_code,
        offload_mode=OFFLOAD_MODE_MODEL_CPU,
        revision=None,
        **kwargs,
    ):
        _require_json_boolean_trust(trust_remote_code)
        raise ValueError(_EXECUTION_UNSUPPORTED_MESSAGE)
