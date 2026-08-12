# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
from collections.abc import Mapping
from copy import deepcopy

from .pipeline_schema import PROTOTYPE_SENSITIVE_FIELD_NAMES
from .pipeline_schema import MoDiffPipelineConfig as PipelineConfig

from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import (
    DEFAULT_GROUP_COMPONENTS,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    apply_component_group_offload,
    configure_components_manager_offload,
    normalize_offload_mode,
    offload_mode_param,
)
from modiff.modular_workflow_discovery import (
    reviewed_modular_workflow_contract,
    select_modular_workflow,
)
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

from . import components
from .custom_pipeline import (
    CUSTOM_PIPELINE_IDENTITY_FIELD,
    CustomPipelineExecutionIdentity,
    resolve_custom_pipeline_binding,
)
from .loaders import record_pipeline_component_runtime_policy, reusable_component_ids
from .modular_utils import require_immutable_hub_revision
from .utils import collect_model_ids


_DECLARATIVE_SIDECAR_ACTIONS = {"show", "hide", "value", "signal"}
logger = logging.getLogger("modiff")


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


def _custom_node_contract(custom_config, workflow_contract=None):
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

    if workflow_contract is not None:
        if "workflow" in sanitized_params:
            raise ValueError("Dynamic Block sidecars must not replace the backend-owned workflow selector.")
        reviewed_workflows = workflow_contract["workflows"]
        if not reviewed_workflows:
            raise ValueError("The reviewed Modular workflow contract has no executable tasks.")
        visibility = {}
        all_workflow_fields = {
            field["name"] for workflow in reviewed_workflows for field in workflow["inputs"]
        }
        all_workflow_outputs = {
            field["name"] for workflow in reviewed_workflows for field in workflow["outputs"]
        }
        unknown_inputs = sorted(set(contract["input_names"]) - all_workflow_fields)
        pipeline_output_names = {
            name[4:] if name.startswith("out_") else name for name in contract["output_names"]
        }
        unknown_outputs = sorted(pipeline_output_names - all_workflow_outputs)
        if unknown_inputs or unknown_outputs:
            details = []
            if unknown_inputs:
                details.append(f"inputs: {', '.join(unknown_inputs)}")
            if unknown_outputs:
                details.append(f"outputs: {', '.join(unknown_outputs)}")
            raise ValueError(
                "Dynamic Block sidecar fields are absent from the reviewed upstream workflow contract ("
                + "; ".join(details)
                + ")."
            )
        sidecar_inputs = set(contract["input_names"])
        workflows = [
            workflow
            for workflow in reviewed_workflows
            if set(workflow["requiredInputs"]).issubset(sidecar_inputs)
        ]
        if not workflows:
            raise ValueError("Dynamic Block sidecar cannot carry the required inputs for any reviewed workflow.")
        for workflow in workflows:
            input_names = {field["name"] for field in workflow["inputs"]}
            output_names = {field["name"] for field in workflow["outputs"]}
            visible = []
            for field_name, definition in sanitized_params.items():
                pipeline_name = field_name[4:] if field_name.startswith("out_") else field_name
                display = definition.get("display")
                if field_name in contract["model_input_names"]:
                    visible.append(field_name)
                elif display == "output" and pipeline_name in output_names:
                    visible.append(field_name)
                elif field_name in input_names:
                    visible.append(field_name)
                elif display != "output" and field_name not in all_workflow_fields:
                    visible.append(field_name)
            visibility[workflow["taskId"]] = visible
        first_task = workflows[0]["taskId"]
        initially_visible = set(visibility[first_task])
        managed_fields = {
            field_name
            for field_name, definition in sanitized_params.items()
            if field_name in all_workflow_fields
            or (
                definition.get("display") == "output"
                and (field_name[4:] if field_name.startswith("out_") else field_name) in all_workflow_outputs
            )
        }
        for field_name in managed_fields:
            sanitized_params[field_name]["hidden"] = field_name not in initially_visible
        sanitized_params = {
            "workflow": {
                "label": "Task",
                "type": "string",
                "value": first_task,
                "options": {workflow["taskId"]: workflow["label"] for workflow in workflows},
                "onChange": {"action": "show", "data": visibility},
                "description": "Tasks and fields come from the reviewed pinned upstream workflow contract.",
            },
            **sanitized_params,
        }
        contract["params"] = sanitized_params
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
            "label": "Repository",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {"noValidation": True, "sources": ["hub"]},
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
            "description": "Repository Python requires a fresh task-scoped authorization and is not restored from workflows.",
        },
        "revision": {
            "label": "Revision",
            "type": "string",
            "value": "",
            "description": "Required immutable 40-character Hugging Face commit hash.",
        },
        "modiff_pipeline_identity": {
            "label": "Reviewed Pipeline Identity",
            "type": "object",
            "value": None,
            "hidden": True,
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
        repo_selector = values.get("repo_id", "")
        if isinstance(repo_selector, Mapping):
            source = repo_selector.get("source")
            repo_id = repo_selector.get("value")
        else:
            source = "hub"
            repo_id = repo_selector
        if source != "hub":
            raise ValueError("Dynamic Block execution requires an immutable Hub repository selection.")
        if not isinstance(repo_id, str) or not repo_id.strip():
            self.send_node_definition({})
            self.set_field_value({CUSTOM_PIPELINE_IDENTITY_FIELD: None})
            return

        trust_remote_code = _require_json_boolean_trust(values.get("trust_remote_code", False))
        if trust_remote_code:
            raise ValueError(
                "Dynamic Block contract preview requires Trust Remote Code off; repository code is not "
                "authorized by this legacy node."
            )
        repo_id = repo_id.strip()
        revision = values.get("revision")
        revision = require_immutable_hub_revision(repo_id, revision, required=True)
        binding = resolve_custom_pipeline_binding(
            source="hub",
            repo_id=repo_id,
            revision=revision,
            trust_remote_code=False,
            expected_identity=None,
        )
        workflow_contract = reviewed_modular_workflow_contract(binding.execution_contract.pipeline_class_name)
        node_config = _custom_node_contract(binding.pipeline_config(), workflow_contract)
        self.set_field_value({CUSTOM_PIPELINE_IDENTITY_FIELD: binding.identity.to_dict()})

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
        modiff_pipeline_identity=None,
        **kwargs,
    ):
        _require_json_boolean_trust(trust_remote_code)
        if trust_remote_code:
            raise ValueError(
                "Repository Python requires a fresh task-scoped operator authorization; imported workflow data "
                "and a persisted trust checkbox are not consent."
            )
        if not isinstance(repo_id, Mapping) or repo_id.get("source") != "hub":
            raise ValueError("Dynamic Block execution requires an explicit immutable Hub repository selection.")
        repository = repo_id.get("value")
        if not isinstance(repository, str) or not repository.strip():
            raise ValueError("Dynamic Block execution requires a non-empty Hub repository ID.")
        if not isinstance(modiff_pipeline_identity, Mapping):
            raise ValueError(
                "Dynamic Block execution requires a backend-issued reviewed identity. Preview the repository first."
            )
        identity = CustomPipelineExecutionIdentity.from_value(modiff_pipeline_identity)
        binding = resolve_custom_pipeline_binding(
            source="hub",
            repo_id=repository.strip(),
            revision=str(revision or "").strip() or None,
            trust_remote_code=False,
            expected_identity=identity,
        )
        offload_mode = normalize_offload_mode(offload_mode, auto_offload=auto_offload, device=device)
        if offload_mode not in {
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        }:
            raise ValueError(f"Dynamic Modular Diffusers blocks do not support {offload_mode} offload.")

        custom_config = binding.pipeline_config()
        workflow_contract = reviewed_modular_workflow_contract(binding.execution_contract.pipeline_class_name)
        node_config = _custom_node_contract(custom_config, workflow_contract)
        available_task_ids = node_config["params"]["workflow"]["options"]
        task_id = kwargs.pop("workflow", next(iter(available_task_ids)))
        if task_id not in available_task_ids:
            raise ValueError(f"Unknown Modular workflow task {task_id!r}, or task unsupported by this sidecar.")
        workflow = select_modular_workflow(workflow_contract, task_id, kwargs)
        pipeline = binding.instantiate(components_manager=components, collection=self.node_id)
        torch_dtype = str_to_dtype(custom_config.default_dtype or "bfloat16")
        configure_components_manager_offload(components, mode=offload_mode, device=device)

        for param_name, param_config in node_config["params"].items():
            if param_name not in kwargs or kwargs[param_name] is None:
                continue
            if param_config.get("type") == "float":
                kwargs[param_name] = float(kwargs[param_name])
            elif param_config.get("type") == "int":
                kwargs[param_name] = int(kwargs[param_name])

        model_input_names = node_config.get("model_input_names", [])
        expected_component_names = pipeline.pretrained_component_names
        model_ids = collect_model_ids(
            kwargs,
            target_key_names=model_input_names,
            target_model_names=expected_component_names,
        )
        components_update_dict = (
            components.get_components_by_ids(ids=model_ids, return_dict_with_names=True) if model_ids else {}
        )
        components_to_load = []
        for component_name in pipeline.pretrained_component_names:
            if component_name in components_update_dict:
                continue
            component_spec = pipeline.get_component_spec(component_name)
            reusable_ids = reusable_component_ids(
                components,
                name=component_name,
                load_id=component_spec.load_id,
                dtype=torch_dtype,
                requested_quantization=None,
                offload_mode=offload_mode,
                device=device,
                node_id=self.node_id,
            )
            if reusable_ids:
                components_update_dict[component_name] = components.get_one(component_id=reusable_ids[0])
            else:
                components_to_load.append(component_name)
        pipeline.update_components(**components_update_dict)
        pipeline.load_components(names=components_to_load, torch_dtype=torch_dtype)

        if offload_mode in {OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK}:
            result = apply_component_group_offload(
                pipeline,
                component_names=DEFAULT_GROUP_COMPONENTS,
                device=device,
                mode=offload_mode,
                node_id=self.node_id,
                scope="dynamic-modular",
            )
            if not result.applied:
                raise RuntimeError("No compatible custom Modular Diffusers component was available to offload.")
        elif offload_mode == OFFLOAD_MODE_NONE:
            pipeline.to(device)
        record_pipeline_component_runtime_policy(
            pipeline,
            offload_mode=offload_mode,
            device=device,
            node_id=self.node_id,
        )

        workflow_input_names = {field["name"] for field in workflow["inputs"]}
        inputs = {
            name: kwargs[name]
            for name in node_config["input_names"]
            if name in workflow_input_names and name in kwargs
        }
        workflow_output_names = {field["name"] for field in workflow["outputs"]}
        output_pairs = [
            (name, name[4:] if name.startswith("out_") else name)
            for name in node_config["output_names"]
            if (name[4:] if name.startswith("out_") else name) in workflow_output_names
        ]
        node_output_names = [name for name, _ in output_pairs]
        pipeline_output_names = [name for _, name in output_pairs]
        pipeline_outputs = pipeline(**inputs, output=pipeline_output_names)
        final_outputs = {
            node_name: pipeline_outputs[pipeline_name]
            for node_name, pipeline_name in zip(node_output_names, pipeline_output_names)
            if pipeline_name in pipeline_outputs
        }
        final_outputs["doc"] = pipeline.blocks.doc
        return final_outputs
