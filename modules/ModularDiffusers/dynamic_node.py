# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging

from diffusers import ModularPipeline
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
from modiff.model_artifact_catalog import resolve_model_revision
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

from . import MESSAGE_DURATION, components
from .loaders import record_pipeline_component_runtime_policy, reusable_component_ids
from .utils import collect_model_ids
from .modular_utils import pin_modular_component_revisions, require_immutable_hub_revision


logger = logging.getLogger("modiff")


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
            "label": "Load Custom Block",
            "display": "ui_button",
            "value": False,
            "onChange": "update_node",
        },
        "device": {"label": "Device", "type": "string", "value": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "auto_offload": {"label": "Enable Auto Offload", "type": "boolean", "value": False},
        "offload_mode": offload_mode_param(modes=[OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK]),
        "trust_remote_code": {"label": "Trust Remote Code", "type": "boolean", "value": False},
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

    def _get_custom_config(self, repo_id, revision=None):
        revision = resolve_model_revision(repo_id, revision)
        revision = require_immutable_hub_revision(repo_id, revision, required=True)
        custom_config = PipelineConfig.load(repo_id, revision=revision)
        return custom_config

    def update_node(self, values, ref):
        if not values.get("repo_id", ""):
            self.send_node_definition({})
            return

        repo_id = values.get("repo_id", "")
        revision = values.get("revision")
        custom_config = self._get_custom_config(repo_id, revision)
        node_config = custom_config.node_params["custom"]

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
        revision = resolve_model_revision(repo_id, revision)
        revision = require_immutable_hub_revision(repo_id, revision, required=True)
        offload_mode = normalize_offload_mode(offload_mode, auto_offload=auto_offload, device=device)
        if offload_mode not in [OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK]:
            self.notify(
                f"Dynamic Modular Diffusers blocks do not support {offload_mode} offload.",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None
        logger.debug(f"Dynamic Block Node ({self.node_id}) received parameters:")
        logger.debug(f"  repo_id: '{repo_id}'")
        logger.debug(f"  device: '{device}'")
        logger.debug(f"  auto_offload: '{auto_offload}'")
        logger.debug(f"  offload_mode: '{offload_mode}'")
        logger.debug(f"  trust_remote_code: '{trust_remote_code}'")

        try:
            pipeline = ModularPipeline.from_pretrained(
                repo_id,
                trust_remote_code=bool(trust_remote_code),
                revision=revision,
                components_manager=components,
                collection=self.node_id,
                local_files_only=True,
            )
        except ValueError as e:
            self.notify(f"{str(e)}", variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            raise e
        except ModuleNotFoundError as e:
            self.notify(
                f"{str(e)}. This likely means the custom code is trying to import a library that is not installed in MoDiff. Please check the error message for which module is missing and install it in your MoDiff environment.",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            raise e

        pin_modular_component_revisions(pipeline, repo_id, revision)

        # Load config to get input/output names and dtype
        custom_config = self._get_custom_config(repo_id, revision)
        node_config = custom_config.node_params["custom"]

        # Get dtype from config
        default_dtype = custom_config.default_dtype
        if not default_dtype:
            default_dtype = "bfloat16"
        torch_dtype = str_to_dtype(default_dtype)

        use_group_offload = offload_mode in [OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK]

        # Configure component-manager residency before component load. CPU and
        # MPS execution deliberately bypass accelerator offload hooks.
        configure_components_manager_offload(components, mode=offload_mode, device=device)

        # Cast parameters to the types expected by the modular pipeline.
        for param_name, param_config in node_config["params"].items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        # Handle components - collect from connected inputs (Load Models) and config
        model_input_names = node_config.get("model_input_names", [])
        expected_component_names = pipeline.pretrained_component_names

        model_ids = collect_model_ids(
            kwargs,
            target_key_names=model_input_names,
            target_model_names=expected_component_names,
        )

        components_update_dict = {}
        if model_ids:
            components_update_dict = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)

        # Check which components need to be loaded vs reused
        components_to_load = []
        for comp_name in pipeline.pretrained_component_names:
            if comp_name in components_update_dict:
                continue  # Already provided externally

            comp_spec = pipeline.get_component_spec(comp_name)
            comp_ids_to_reuse = reusable_component_ids(
                components,
                name=comp_name,
                load_id=comp_spec.load_id,
                dtype=torch_dtype,
                requested_quantization=None,
                offload_mode=offload_mode,
                device=device,
                node_id=self.node_id,
            )
            if comp_ids_to_reuse:
                # Reuse existing component
                comp_id = comp_ids_to_reuse[0]
                components_update_dict[comp_name] = components.get_one(component_id=comp_id)
            else:
                components_to_load.append(comp_name)

        pipeline.update_components(**components_update_dict)
        pipeline.load_components(names=components_to_load, torch_dtype=torch_dtype)

        if use_group_offload:
            try:
                offload_result = apply_component_group_offload(
                    pipeline,
                    component_names=DEFAULT_GROUP_COMPONENTS,
                    device=device,
                    mode=offload_mode,
                    node_id=self.node_id,
                    scope="dynamic-modular",
                )
                if not offload_result.applied:
                    raise RuntimeError("No compatible custom Modular Diffusers component was available to offload.")
                logger.debug(f"Dynamic Block Node: applied {offload_mode} to {offload_result.components}")
            except RuntimeError as exc:
                self.notify(str(exc), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
                raise
        elif offload_mode == "none":
            pipeline.to(device)

        record_pipeline_component_runtime_policy(
            pipeline,
            offload_mode=offload_mode,
            device=device,
            node_id=self.node_id,
        )

        # Build inputs dict
        inputs_dict = {}
        for input_name in node_config["input_names"]:
            if input_name in kwargs:
                inputs_dict[input_name] = kwargs.pop(input_name)

        # Execute pipeline - strip out_ prefix for pipeline call
        node_output_names = node_config["output_names"]
        pipeline_output_names = [name[4:] if name.startswith("out_") else name for name in node_output_names]
        pipeline_outputs = pipeline(**inputs_dict, output=pipeline_output_names)

        # Map pipeline outputs back to node names (with the out_ prefix).
        final_outputs = {}
        for node_name, pipeline_name in zip(node_output_names, pipeline_output_names):
            if pipeline_name in pipeline_outputs:
                final_outputs[node_name] = pipeline_outputs[pipeline_name]

        final_outputs["doc"] = pipeline.blocks.doc
        return final_outputs
