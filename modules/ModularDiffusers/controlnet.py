# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging

from modiff.NodeBase import NodeBase

from . import components
from .modular_utils import (
    modular_generator_from_seed,
    normalize_modular_runtime_params,
    normalize_modular_seed,
    pipeline_class_from_model_type,
    pipeline_class_from_runtime_inputs,
    reject_undeclared_modular_generator,
    require_modiff_node_contract,
)
from .route_state import (
    ROUTE_RESERVED_PIPELINE_INPUTS,
    ROUTE_STATE_INPUT,
    ROUTE_STATE_OUTPUT,
    SDXL_UNION_CONTROL_MODE_LIMIT,
    consume_controlnet_input_route_state,
    issue_controlnet_route_state,
    reject_route_reserved_inputs,
    reject_route_reserved_inputs_before_identity_resolution,
    require_component_binding,
    require_route_state_current_publication,
    require_route_state_shape_before_identity_resolution,
    require_sdxl_controlnet_component_binding,
    require_standalone_component_binding,
    route_cache_params_equal,
    route_contract_for_model_type,
    validate_controlnet_input_route_state,
    validate_route_field_contract,
)
from .utils import collect_model_ids


logger = logging.getLogger("modiff")

_SDXL_CONTROLNET_VARIANTS = frozenset({"ordinary", "union"})


def _sdxl_controlnet_selection(values):
    variant = values.get("controlnet_variant", "ordinary")
    if type(variant) is not str or variant not in _SDXL_CONTROLNET_VARIANTS:
        raise ValueError("SDXL ControlNet variant must be exactly 'ordinary' or 'union'.")
    if variant == "ordinary":
        return False, None
    control_mode = values.get("control_mode")
    if type(control_mode) is not int or not 0 <= control_mode < SDXL_UNION_CONTROL_MODE_LIMIT:
        raise ValueError("SDXL ControlNet Union requires one bounded control-type index.")
    return True, control_mode


class ControlnetUnion(NodeBase):
    label = "ControlNet Union"
    category = "adapters"

    params = {
        "pose_image": {
            "label": "Pose image",
            "type": "image",
            "display": "input",
        },
        "depth_image": {
            "label": "Depth image",
            "type": "image",
            "display": "input",
        },
        "edges_image": {
            "label": "Edges image",
            "type": "image",
            "display": "input",
        },
        "lines_image": {
            "label": "Lines image",
            "type": "image",
            "display": "input",
        },
        "normal_image": {
            "label": "Normal image",
            "type": "image",
            "display": "input",
        },
        "segment_image": {
            "label": "Segment image",
            "type": "image",
            "display": "input",
        },
        "tile_image": {
            "label": "Tile image",
            "type": "image",
            "display": "input",
        },
        "repaint_image": {
            "label": "Repaint image",
            "type": "image",
            "display": "input",
        },
        "controlnet_conditioning_scale": {
            "label": "Scale",
            "type": "float",
            "display": "slider",
            "default": 0.5,
            "min": 0,
            "max": 1,
        },
        "control_guidance_start": {
            "label": "Start",
            "type": "float",
            "display": "slider",
            "default": 0.0,
            "min": 0,
            "max": 1,
        },
        "control_guidance_end": {
            "label": "End",
            "type": "float",
            "display": "slider",
            "default": 1.0,
            "min": 0,
            "max": 1,
        },
        "controlnet_model": {
            "label": "Controlnet Union Model",
            "type": "diffusers_auto_model",
            "display": "input",
        },
        "controlnet": {
            "label": "Controlnet",
            "display": "output",
            "type": "custom_controlnet",
        },
    }

    def execute(
        self,
        pose_image,
        depth_image,
        edges_image,
        lines_image,
        normal_image,
        segment_image,
        tile_image,
        repaint_image,
        controlnet_conditioning_scale,
        controlnet_model,
        control_guidance_start,
        control_guidance_end,
    ):
        image_map = {
            "pose_image": (pose_image, 0),
            "depth_image": (depth_image, 1),
            "edges_image": (edges_image, 2),
            "lines_image": (lines_image, 3),
            "normal_image": (normal_image, 4),
            "segment_image": (segment_image, 5),
            "tile_image": (tile_image, 6),
            "repaint_image": (repaint_image, 7),
        }

        control_mode = []
        control_image = []

        for key, (image, index) in image_map.items():
            if image is not None:
                control_mode.append(index)
                control_image.append(image)

        controlnet = {
            "controlnet_model": controlnet_model,
            "controlnet_inputs": {
                "control_image": control_image,
                "control_mode": control_mode,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            },
        }
        return {"controlnet": controlnet}


class Controlnet(NodeBase):
    label = "ControlNet"
    category = "adapters"
    resizable = True
    skipParamsCheck = True
    node_type = "controlnet"

    params = {
        "model_type": {"label": "Model Type", "type": "string", "default": "", "hidden": True},
        "controlnet_bundle": {
            "label": "Controlnet",
            "display": "output",
            "type": "custom_controlnet",
            "onSignal": [
                {"action": "value", "target": "model_type"},
                {"action": "exec", "data": "update_node"},
            ],
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def update_node(self, values, ref):
        node_params = {
            "model_type": {
                "label": "Model Type",
                "type": "string",
                "default": "",
                "hidden": True,
            },
            "controlnet_bundle": {
                "label": "Controlnet",
                "display": "output",
                "type": "custom_controlnet",
                "onSignal": [
                    {"action": "value", "target": "model_type"},
                    {"action": "exec", "data": "update_node"},
                ],
            },
        }

        model_type = values.get("model_type", "")

        if self._model_type == model_type:
            if not model_type or self._pipeline_class is None:
                return None
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                require_blocks=False,
                resolve_blocks=False,
            )
            node_params_to_update = dict(node_config["params"])
            node_params_to_update.pop("controlnet_bundle", None)
            self.send_node_definition({**node_params, **node_params_to_update})
            return None
        if model_type == "":
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            return None

        try:
            self._pipeline_class = pipeline_class_from_model_type(model_type)
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                require_blocks=False,
                resolve_blocks=False,
            )
        except ValueError:
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            raise
        self._model_type = model_type

        node_params_to_update = node_config["params"]
        node_params_to_update.pop("controlnet_bundle", None)

        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def _cache_params_equal(self, previous, current):
        equal = route_cache_params_equal(previous, current, fallback=super()._cache_params_equal)
        if not equal or not isinstance(current, dict):
            return equal

        route_state = current.get(ROUTE_STATE_INPUT)
        vae = current.get("vae")
        if route_contract_for_model_type(self._model_type) == "sdxl":
            require_route_state_shape_before_identity_resolution(current)
            reject_undeclared_modular_generator(current)
            reject_route_reserved_inputs_before_identity_resolution(current)
            reject_route_reserved_inputs(current, model_type=self._model_type, action=self.node_type)
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                require_blocks=False,
                resolve_blocks=False,
            )
            validate_route_field_contract(current, node_config)
            union, _control_mode = _sdxl_controlnet_selection(current)
            if {"control_type", "control_type_idx"}.intersection(current):
                raise ValueError("SDXL ControlNet Union output fields are backend-managed.")
            require_sdxl_controlnet_component_binding(current.get("controlnet"), union=union)
            return True
        routed_action = route_state is not None or self._model_type == "QwenImageModularPipeline"
        routed_action = routed_action or (
            isinstance(vae, dict) and vae.get("model_type") == "QwenImageModularPipeline"
        )
        if not routed_action:
            return True

        require_route_state_shape_before_identity_resolution(current)
        reject_undeclared_modular_generator(current)
        reject_route_reserved_inputs_before_identity_resolution(current)
        reject_route_reserved_inputs(current)
        require_route_state_current_publication(route_state, label="ControlNet input route")
        seed = normalize_modular_seed(current.get("seed"))
        binding = require_component_binding(
            vae,
            label="ControlNet VAE",
            expected_model_type="QwenImageModularPipeline",
            expected_role="vae",
        )
        require_standalone_component_binding(
            current.get("controlnet"),
            label="ControlNet model",
            expected_kind="controlnet",
        )
        if route_state is not None:
            validate_controlnet_input_route_state(
                route_state,
                binding=binding,
                model_type="QwenImageModularPipeline",
                seed=seed,
            )
        return True

    def execute(self, **kwargs):
        kwargs = dict(kwargs)
        require_route_state_shape_before_identity_resolution(kwargs)
        reject_undeclared_modular_generator(kwargs)
        reject_route_reserved_inputs_before_identity_resolution(kwargs)
        identity_kwargs = {name: value for name, value in kwargs.items() if name != ROUTE_STATE_INPUT}
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, identity_kwargs)
        self._model_type = getattr(self._pipeline_class, "__name__", "")

        # 1. Get node config
        blocks, node_config = require_modiff_node_contract(
            self._pipeline_class,
            self.node_type,
            require_blocks=False,
        )
        denoise_blocks, _ = require_modiff_node_contract(self._pipeline_class, "denoise")
        validate_route_field_contract(kwargs, node_config)
        if route_contract_for_model_type(self._model_type) == "sdxl":
            union, control_mode = _sdxl_controlnet_selection(kwargs)
            if {"control_type", "control_type_idx"}.intersection(kwargs):
                raise ValueError("SDXL ControlNet Union output fields are backend-managed.")
            require_sdxl_controlnet_component_binding(kwargs.get("controlnet"), union=union)
            kwargs.pop("controlnet_variant", None)
            kwargs.pop("control_mode", None)
            if union:
                kwargs["control_mode"] = control_mode

        route_output_declared = ROUTE_STATE_OUTPUT in node_config["output_names"]
        route_state = kwargs.get(ROUTE_STATE_INPUT)
        route_binding = None
        controlnet_binding = None
        if route_output_declared:
            if blocks is None or "generator" not in blocks.input_names:
                raise ValueError("The routed Modular ControlNet block does not expose its required generator input.")
            kwargs = normalize_modular_runtime_params(kwargs, node_config)
            reject_route_reserved_inputs(kwargs)
            if kwargs.get("seed") is None:
                raise ValueError("A routed Modular ControlNet action requires its originating seed.")
            route_binding = require_component_binding(
                kwargs.get("vae"),
                label="ControlNet VAE",
                expected_model_type=self._model_type,
                expected_role="vae",
            )
            controlnet_binding = require_standalone_component_binding(
                kwargs.get("controlnet"),
                label="ControlNet model",
                expected_kind="controlnet",
            )
            if route_state is not None:
                validate_controlnet_input_route_state(
                    route_state,
                    binding=route_binding,
                    model_type=self._model_type,
                    seed=kwargs["seed"],
                )
        elif route_state is not None:
            raise ValueError(f"Pipeline '{self._model_type}' does not support opaque Modular route state.")

        # 2. Cast parameters to the types expected by the modular pipeline.
        # Preserve the graph compatibility cast until the upstream schema exposes exact types.
        for param_name, param_config in node_config["params"].items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        node_output = None
        if blocks is not None:
            # 3. Create pipeline
            self._pipeline = blocks.init_pipeline(components_manager=components)

            def revalidate_route_components():
                if not route_output_declared:
                    return
                require_component_binding(
                    kwargs.get("vae"),
                    label="ControlNet VAE",
                    expected_model_type=self._model_type,
                    expected_token=route_binding,
                    expected_role="vae",
                )
                require_standalone_component_binding(
                    kwargs.get("controlnet"),
                    label="ControlNet model",
                    expected_kind="controlnet",
                    expected_binding=controlnet_binding,
                )

            # Pipeline initialization and component-manager activity are
            # extension boundaries.  Recheck both publications before manager
            # lookup and again before the upstream block can execute.
            revalidate_route_components()

            # 4. Update components
            expected_component_names = blocks.component_names
            model_input_names = node_config["model_input_names"]
            model_ids = collect_model_ids(
                kwargs,
                target_key_names=model_input_names,
                target_model_names=expected_component_names,
            )

            if model_ids:
                components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
                if components_to_update:
                    self._pipeline.update_components(**components_to_update)

            revalidate_route_components()

            if route_output_declared:
                if route_state is None:
                    route_generator = modular_generator_from_seed(kwargs["seed"], self._pipeline)
                else:
                    route_generator = consume_controlnet_input_route_state(
                        route_state,
                        binding=route_binding,
                        model_type=self._model_type,
                        seed=kwargs["seed"],
                        execution_device=self._pipeline._execution_device,
                    )

            # 5. Compile runtime inputs from kwargs based on node_config.inputs
            node_kwargs = {}
            input_names = node_config["input_names"]

            for name in input_names:
                if name not in kwargs:
                    continue
                value = kwargs.get(name)

                if name in {ROUTE_STATE_INPUT, "seed"}:
                    continue

                if isinstance(value, dict) and name not in blocks.input_names:
                    for k, v in value.items():
                        if k in blocks.input_names:
                            node_kwargs[k] = v
                elif name in blocks.input_names:
                    node_kwargs[name] = value

            if route_output_declared:
                node_kwargs["generator"] = route_generator

            # 6. Run the pipeline
            revalidate_route_components()
            node_output = self._pipeline(**node_kwargs).values
            # The upstream block call is also an extension boundary.  Do not
            # publish route state if it changed either reviewed component
            # payload while producing the control latents.
            revalidate_route_components()

        # 7. Prepare controlnet output for Denoise node
        # use the denoise blocks to know what inputs it expects

        controlnet_inputs = {}
        for name in denoise_blocks.input_names:
            # Qwen's upstream control encoder returns the generator in its
            # PipelineState even though the public block output is only the
            # control latents.  The opaque route state below is the sole
            # trusted carrier for that advanced generator (and other
            # route-owned values); never duplicate it into a graph bundle.
            if route_output_declared and name in ROUTE_RESERVED_PIPELINE_INPUTS:
                continue
            if node_output and name in node_output and node_output[name] is not None:
                controlnet_inputs.update({name: node_output[name]})
            elif name in kwargs and kwargs[name] is not None:
                controlnet_inputs.update({name: kwargs.pop(name)})

        controlnet_out = {
            "controlnet": kwargs.get("controlnet"),
            **controlnet_inputs,
        }

        outputs = {"controlnet_bundle": controlnet_out}
        if route_output_declared:
            outputs[ROUTE_STATE_OUTPUT] = issue_controlnet_route_state(
                route_state,
                binding=route_binding,
                controlnet_component=kwargs.get("controlnet"),
                seed=kwargs["seed"],
                generator=node_kwargs["generator"],
                control_image_latents=node_output.get("control_image_latents") if node_output else None,
            )
        return outputs
