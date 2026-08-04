# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
import traceback
from collections.abc import Mapping

import torch
from diffusers import ComponentSpec, ModularPipeline
from diffusers.utils import logging as diffusers_logging
from .pipeline_schema import MoDiffPipelineConfig as PipelineConfig

from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import (
    DEFAULT_GROUP_COMPONENTS,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    apply_component_group_offload,
    apply_model_offload,
    configure_components_manager_offload,
    normalize_offload_mode,
    offload_mode_param,
)
from modiff.model_artifact_catalog import resolve_model_revision
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

from . import MESSAGE_DURATION, MODULAR_REGISTRY, components
from .modular_utils import (
    DUMMY_CUSTOM_PIPELINE_CONFIG,
    DummyCustomPipeline,
    get_all_model_types,
    get_model_type_metadata,
    pin_modular_component_revisions,
    require_immutable_hub_revision,
)


logger = logging.getLogger("modiff")
logger.setLevel(logging.DEBUG)

QWEN_LOW_VRAM_COMPONENT = "qwen_low_vram"
QWEN_LOW_RESOURCE_COMPONENTS = {"transformer", "text_encoder"}
GROUP_OFFLOAD_COMPONENTS = set(DEFAULT_GROUP_COMPONENTS)


def node_get_component_info(node_id=None, manager=None, name=None):
    comp_ids = manager._lookup_ids(name=name, collection=node_id)
    if len(comp_ids) != 1:
        raise ValueError(f"Expected 1 component for {name} for node {node_id}, got {len(comp_ids)}")
    return manager.get_model_info(list(comp_ids)[0])


def quant_config_to_info(config):
    if hasattr(config, "to_diff_dict"):
        return config.to_diff_dict()
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return str(config)


def component_quant_config_summary(config):
    if not config:
        return {}
    return {name: quant_config_to_info(value) for name, value in config.items()}


def should_incrementally_group_offload(*, use_group_offload, quant_config):
    """Select the low-peak loader path from component capabilities, not a pipeline name."""
    return bool(
        use_group_offload and quant_config and QWEN_LOW_RESOURCE_COMPONENTS.intersection(set(quant_config.keys()))
    )


def safe_diagnostic_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_diagnostic_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_diagnostic_value(item) for key, item in value.items()}
    return str(value)


class RequiredComponentLoadError(RuntimeError):
    def __init__(self, component_name, model_id, dtype, offload_mode, quantization, original_error, traceback_text):
        self.component_name = component_name
        self.model_id = model_id
        self.dtype = str(dtype)
        self.offload_mode = str(offload_mode)
        self.quantization = safe_diagnostic_value(quantization)
        self.original_error = original_error
        self.traceback_text = traceback_text
        quantized_components = (
            ", ".join(sorted(self.quantization.keys()))
            if isinstance(self.quantization, dict)
            else str(self.quantization)
        )
        super().__init__(
            f"Required Diffusers component '{component_name}' failed to load for {model_id} "
            f"(dtype={self.dtype}, offload={self.offload_mode}, quantized={quantized_components or 'none'}): "
            f"{original_error}"
        )


def component_load_kwargs_for(name, kwargs):
    component_load_kwargs = {}
    for key, value in kwargs.items():
        if not isinstance(value, dict):
            component_load_kwargs[key] = value
        elif name in value:
            component_load_kwargs[key] = value[name]
        elif "default" in value:
            component_load_kwargs[key] = value["default"]
    return component_load_kwargs


def place_pipeline_components(pipeline, device, progress_callback=None):
    """Place resident model components one at a time with truthful progress.

    Diffusers' pipeline-level ``to`` call iterates these same modules but gives
    callers no indication which multi-gigabyte component is being copied. On
    unified-memory accelerators an individual copy can take minutes, so retain
    the normal module placement semantics while exposing the component name.
    """

    resident_components = [
        (name, component) for name, component in pipeline.components.items() if isinstance(component, torch.nn.Module)
    ]
    total = len(resident_components)
    for index, (name, component) in enumerate(resident_components, start=1):
        if progress_callback:
            progress_callback(name, index, total)
        component.to(device)
    return [name for name, _component in resident_components]


def component_reuse_compatible(
    component,
    *,
    dtype,
    requested_quantization,
    offload_mode,
    device,
    node_id=None,
):
    """Return whether a shared component matches the complete runtime policy."""

    if not isinstance(component, torch.nn.Module):
        return True

    component_dtype = getattr(component, "dtype", None)
    if component_dtype is None:
        try:
            component_dtype = next(component.parameters()).dtype
        except StopIteration:
            component_dtype = None
    if component_dtype != dtype:
        return False

    existing_quantizer = getattr(component, "hf_quantizer", None)
    if requested_quantization is None:
        if existing_quantizer is not None:
            return False
    elif existing_quantizer is None:
        return False
    else:
        existing_config = getattr(existing_quantizer, "quantization_config", None)
        existing_info = quant_config_to_info(existing_config) if existing_config is not None else None
        if existing_info != quant_config_to_info(requested_quantization):
            return False

    target_device = str(torch.device(device))
    recorded_mode = getattr(component, "_modiff_offload_mode", None)
    recorded_device = getattr(component, "_modiff_execution_device", None)
    if recorded_mode is not None or recorded_device is not None:
        if recorded_mode != offload_mode or recorded_device != target_device:
            return False
        if offload_mode == OFFLOAD_MODE_GROUP_DISK:
            recorded_node_id = getattr(component, "_modiff_offload_node_id", None)
            if node_id is None or recorded_node_id != str(node_id):
                return False
        return True

    # Legacy resident components have no explicit policy metadata. Their
    # current device is enough to prove compatibility only for a hook-free
    # resident run; never infer compatibility for an offloaded component.
    if offload_mode != OFFLOAD_MODE_NONE:
        return False
    try:
        component_device = torch.device(component.device)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        try:
            component_device = next(component.parameters()).device
        except StopIteration:
            return False
    return component_device == torch.device(device)


def reusable_component_ids(
    manager,
    *,
    name,
    load_id,
    dtype,
    requested_quantization,
    offload_mode,
    device,
    node_id=None,
):
    """Find deterministic, name-scoped shared components safe for this run."""

    if not load_id or load_id == "null":
        return []
    compatible_ids = []
    for component_id in sorted(manager._lookup_ids(name=name, load_id=load_id)):
        component = manager.get_one(component_id=component_id)
        if component_reuse_compatible(
            component,
            dtype=dtype,
            requested_quantization=requested_quantization,
            offload_mode=offload_mode,
            device=device,
            node_id=node_id,
        ):
            compatible_ids.append(component_id)
    return compatible_ids


def record_pipeline_component_runtime_policy(pipeline, *, offload_mode, device, node_id=None):
    """Annotate model components after their placement/hooks have been applied."""

    try:
        pipeline_components = pipeline.components
    except (AttributeError, RuntimeError):
        pipeline_components = {}
    for component in pipeline_components.values():
        if not isinstance(component, torch.nn.Module):
            continue
        component._modiff_offload_mode = offload_mode
        component._modiff_execution_device = str(torch.device(device))
        component._modiff_offload_node_id = str(node_id) if offload_mode == OFFLOAD_MODE_GROUP_DISK else None


def reusable_standalone_component(
    manager,
    *,
    name,
    load_id,
    dtype,
    offload_mode,
    device,
    node_id=None,
):
    """Return a compatible resident standalone model, if one already exists."""

    if not load_id or load_id == "null":
        return None
    for component_id in sorted(manager._lookup_ids(name=name, load_id=load_id)):
        component = manager.get_one(component_id=component_id)
        if not isinstance(component, torch.nn.Module):
            continue
        if component_reuse_compatible(
            component,
            dtype=dtype,
            requested_quantization=None,
            offload_mode=offload_mode,
            device=device,
            node_id=node_id,
        ):
            return component_id, component

    return None


def load_components_strict(
    pipeline,
    names,
    *,
    required_names,
    model_id,
    dtype,
    offload_mode,
    quant_config,
    diagnostics,
    component_load_kwargs,
    incremental_group_offload=None,
):
    if names is None:
        names = [
            name
            for name in pipeline._component_specs.keys()
            if pipeline._component_specs[name].default_creation_method == "from_pretrained"
            and pipeline._component_specs[name].pretrained_model_name_or_path is not None
            and getattr(pipeline, name, None) is None
        ]
    elif isinstance(names, str):
        names = [names]
    elif not isinstance(names, list):
        raise ValueError(f"Invalid type for names: {type(names)}")

    required_names = set(required_names or [])
    components_to_load = [name for name in names if name in pipeline._component_specs]
    unknown_names = [name for name in names if name not in pipeline._component_specs]
    if unknown_names:
        diagnostics.setdefault("components_unknown", []).extend(unknown_names)
        if required_names.intersection(unknown_names):
            missing = sorted(required_names.intersection(unknown_names))
            raise RuntimeError(f"Required Diffusers component specs are missing: {', '.join(missing)}")
        logger.warning("Unknown components will be ignored: %s", unknown_names)

    # Keep an explicit outer component bar around individually loaded modular
    # components. Nested Diffusers/Transformers shard and weight bars then
    # inherit the component rank, while a component with a silent placement
    # phase still leaves a truthful "which component" status in the queue.
    for name in diffusers_logging.tqdm(
        components_to_load,
        desc="Loading model components",
        disable=True,
    ):
        spec = pipeline._component_specs[name]
        load_kwargs = component_load_kwargs_for(name, component_load_kwargs)
        if (
            "trust_remote_code" in load_kwargs
            and getattr(pipeline, "_pretrained_model_name_or_path", None) is not None
            and spec.pretrained_model_name_or_path != pipeline._pretrained_model_name_or_path
        ):
            load_kwargs.pop("trust_remote_code", None)

        if not spec.pretrained_model_name_or_path:
            diagnostics.setdefault("components_skipped", []).append(
                {"name": name, "reason": "no pretrained model path"}
            )
            continue

        try:
            component = spec.load(**load_kwargs)
        except Exception as exc:
            traceback_text = traceback.format_exc()
            failure = {
                "name": name,
                "required": name in required_names,
                "error": str(exc) or type(exc).__name__,
                "traceback": traceback_text,
                "load_kwargs": safe_diagnostic_value(load_kwargs),
            }
            diagnostics.setdefault("components_failed", []).append(failure)
            if name in required_names:
                raise RequiredComponentLoadError(
                    name,
                    model_id,
                    dtype,
                    offload_mode,
                    component_quant_config_summary(quant_config),
                    exc,
                    traceback_text,
                ) from exc
            logger.warning("Optional Diffusers component %s failed to load: %s", name, exc)
            continue

        pipeline.register_components(**{name: component})
        diagnostics.setdefault("components_loaded", []).append(name)

        if incremental_group_offload and name in GROUP_OFFLOAD_COMPONENTS:
            result = apply_component_group_offload(
                pipeline,
                component_names=[name],
                device=incremental_group_offload["device"],
                mode=incremental_group_offload["mode"],
                node_id=incremental_group_offload["node_id"],
                scope=incremental_group_offload["scope"],
            )
            if result.applied:
                offload_diag = diagnostics.setdefault("offload", {})
                offload_diag["incremental"] = True
                offload_diag["method"] = result.method
                offload_diag["mode"] = result.mode
                offload_diag["components"] = sorted(set(offload_diag.get("components", []) + result.components))
                if result.disk_path:
                    offload_diag["disk_path"] = result.disk_path
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def normalize_quant_config_input(config):
    if config is None:
        return None
    if isinstance(config, str):
        if config.strip() == "":
            return None
        raise TypeError(
            "Quant Config must be connected to a Quantization Config node output. "
            f"Received unresolved string value {config!r}."
        )
    if hasattr(config, "quant_mapping"):
        config = config.quant_mapping
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(
        "Quant Config must be a component-to-config mapping from Quantization Config node. "
        f"Received {type(config).__name__}."
    )


def update_lora_adapters(lora_node, lora_list):
    """
    Update LoRA adapters based on the provided list of LoRAs.

    Args:
        lora_node: ModularPipeline node containing LoRA functionality
        lora_list: List of dictionaries or single dictionary containing LoRA configurations with:
                  {'lora_path': str, 'weight_name': str, 'adapter_name': str, 'scale': float}
    """
    # Convert single lora to list if needed
    if not isinstance(lora_list, list):
        lora_list = [lora_list]

    # Get currently loaded adapters
    loaded_adapters = list(set().union(*lora_node.get_list_adapters().values()))

    # Determine which adapters to set and remove
    to_set = [lora["adapter_name"] for lora in lora_list]
    to_remove = [adapter for adapter in loaded_adapters if adapter not in to_set]

    # Remove unused adapters first
    for adapter_name in to_remove:
        lora_node.delete_adapters(adapter_name)

    # Load new LoRAs and set their scales
    scales = {}
    for lora in lora_list:
        adapter_name = lora["adapter_name"]
        if adapter_name not in loaded_adapters:
            lora_node.load_lora_weights(
                lora["lora_path"],
                weight_name=lora["weight_name"],
                adapter_name=adapter_name,
            )
        scales[adapter_name] = lora["scale"]

    # Set adapter scales
    if scales:
        lora_node.set_adapters(list(scales.keys()), list(scales.values()))


def apply_lora_scheduler_override(pipeline, lora_list):
    """Apply one explicit scheduler contract supplied by distilled LoRAs."""
    if not isinstance(lora_list, list):
        lora_list = [lora_list]
    overrides = [
        (item.get("scheduler_class"), item.get("scheduler_config") or {})
        for item in lora_list
        if item.get("scheduler_class")
    ]
    if not overrides:
        return None
    if any(override != overrides[0] for override in overrides[1:]):
        raise ValueError("Connected LoRAs declare incompatible scheduler contracts.")

    scheduler_class_name, scheduler_config = overrides[0]
    scheduler_class = getattr(__import__("diffusers", fromlist=[scheduler_class_name]), scheduler_class_name)
    current_scheduler = getattr(pipeline, "scheduler", None)
    if current_scheduler is None:
        raise ValueError("The selected LoRA requires a scheduler, but the pipeline does not expose one.")
    scheduler = scheduler_class.from_config(current_scheduler.config, **scheduler_config)
    pipeline.update_components(scheduler=scheduler)
    return scheduler


class QuantizationConfigNode(NodeBase):
    label = "Quantization Config"
    category = "loader"
    resizable = True
    skipParamsCheck = True

    params = {
        "model_id": {
            "label": "Model ID",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
            },
        },
        "subfolder": {
            "label": "Subfolder",
            "type": "string",
            "value": "transformer",
        },
        "load_layers_button": {
            "label": "Load Model Layers",
            "display": "ui_button",
            "value": False,
            "onChange": "update_skip_modules",
        },
        "component": {
            "label": "Component",
            "type": "string",
            "options": ["transformer", "text_encoder", "qwen_low_vram"],
            "value": "transformer",
        },
        "quant_type": {
            "label": "Quant Type",
            "type": "string",
            "options": ["bnb_4bit", "bnb_8bit"],
            "value": "bnb_4bit",
            "onChange": {
                "bnb_4bit": ["bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant"],
                "bnb_8bit": ["llm_int8_threshold", "llm_int8_has_fp16_weight"],
            },
        },
        "bnb_4bit_quant_type": {
            "label": "4-bit Quant Type",
            "type": "string",
            "options": ["nf4", "fp4"],
            "value": "nf4",
        },
        "bnb_4bit_compute_dtype": {
            "label": "Compute Dtype",
            "type": "string",
            "options": ["", "float32", "float16", "bfloat16"],
            "value": "",
        },
        "bnb_4bit_use_double_quant": {
            "label": "Double Quant",
            "type": "boolean",
            "value": False,
        },
        "llm_int8_threshold": {
            "label": "Int8 Threshold",
            "type": "float",
            "display": "slider",
            "default": 6.0,
            "min": 0.0,
            "max": 10.0,
            "step": 0.5,
        },
        "llm_int8_has_fp16_weight": {
            "label": "Has FP16 Weight",
            "type": "boolean",
            "value": False,
        },
        "llm_int8_skip_modules": {
            "label": "Skip Modules",
            "type": "string",
            "display": "select",
            "options": [],
            "fieldOptions": {"multiple": True},
            "value": [],
        },
        "quantization_config": {
            "label": "Quantization Config",
            "type": "quant_config",
            "display": "output",
        },
        "config_info": {
            "label": "Quantization Config Info",
            "type": "string",
            "display": "output",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._cached_layers = {}

    def _get_model_layers(self, model_id, subfolder):
        """Load model with empty weights and extract block-level layer groups."""
        cache_key = f"{model_id}|{subfolder}"
        if cache_key in self._cached_layers:
            return self._cached_layers[cache_key]

        try:
            import torch.nn as nn
            from accelerate import init_empty_weights
            from diffusers import AutoModel

            config = AutoModel.load_config(model_id, subfolder=subfolder)

            if "_class_name" not in config:
                raise ValueError(f"Config at {model_id}/{subfolder} doesn't contain '_class_name'")

            with init_empty_weights():
                # AutoModel is the public model boundary. Reaching into the
                # standard-pipeline loader internals from a Modular adapter
                # couples the two pipeline systems and breaks the upstream
                # separation contract.
                model = AutoModel.from_config(config)

            # Get all Linear layer names
            linear_layers = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

            # Group by first two parts: e.g. "transformer_blocks.0.attn.to_q" -> "transformer_blocks.0"
            blocks = {}
            for layer_name in linear_layers:
                parts = layer_name.split(".")
                block_prefix = ".".join(parts[:2]) if len(parts) > 2 else layer_name

                if block_prefix not in blocks:
                    blocks[block_prefix] = []
                blocks[block_prefix].append(layer_name)

            del model
            self._cached_layers[cache_key] = blocks
            return blocks

        except Exception as e:
            logger.warning(f"Failed to load model layers from {model_id}/{subfolder}: {e}")
            return {}

    def update_skip_modules(self, values, ref):
        """Called when 'Load Model Layers' button is clicked."""
        model_id = values.get("model_id", {})
        if isinstance(model_id, dict):
            model_id = model_id.get("value", "")

        subfolder = values.get("subfolder", "")

        if not model_id:
            self.notify(
                "Please enter a Model ID first.",
                variant="warning",
                persist=False,
                autoHideDuration=3000,
            )
            return

        self.notify(
            f"Loading layers from {model_id}...",
            variant="info",
            persist=False,
            autoHideDuration=2000,
        )

        blocks = self._get_model_layers(model_id, subfolder)

        if blocks:
            self.set_field_params(
                "llm_int8_skip_modules",
                {
                    "options": list(blocks.keys()),
                    "value": [],
                },
            )
            self.notify(
                f"Loaded {len(blocks)} blocks/layers.",
                variant="success",
                persist=False,
                autoHideDuration=3000,
            )
        else:
            self.notify(
                "No layers found or failed to load model.",
                variant="error",
                persist=False,
                autoHideDuration=3000,
            )

    def execute(
        self,
        model_id,
        subfolder,
        component,
        quant_type,
        bnb_4bit_quant_type,
        bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant,
        llm_int8_threshold,
        llm_int8_has_fp16_weight,
        llm_int8_skip_modules,
        **kwargs,
    ):
        import torch
        from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
        from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

        def str_to_dtype(dtype_str):
            dtype_map = {
                "": None,
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            return dtype_map.get(dtype_str, None)

        def create_quant_config(config_cls, skip=None):
            if quant_type == "bnb_4bit":
                return config_cls(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=str_to_dtype(bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    llm_int8_skip_modules=skip,
                )
            if quant_type == "bnb_8bit":
                return config_cls(
                    load_in_8bit=True,
                    llm_int8_threshold=float(llm_int8_threshold),
                    llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
                    llm_int8_skip_modules=skip,
                )
            raise ValueError(f"Unsupported quantization type: {quant_type}")

        skip_modules = llm_int8_skip_modules if llm_int8_skip_modules else None

        if skip_modules:
            if isinstance(model_id, dict):
                model_id = model_id.get("value", "")
            cache_key = f"{model_id}|{subfolder}"
            blocks = self._cached_layers.get(cache_key, {})
            if blocks:
                block_keys = list(blocks.keys())
                # Resolve indices to block names, then expand to all linear layers in those blocks
                resolved = []
                for i in skip_modules:
                    idx = int(i)
                    if idx < len(block_keys):
                        resolved.extend(blocks[block_keys[idx]])
                skip_modules = resolved

        if component == QWEN_LOW_VRAM_COMPONENT:
            if quant_type != "bnb_4bit":
                raise ValueError("Qwen low-VRAM bundle currently supports only bnb_4bit quantization.")
            quantization_config = {
                "transformer": create_quant_config(DiffusersBitsAndBytesConfig, skip_modules),
                "text_encoder": create_quant_config(TransformersBitsAndBytesConfig),
            }
        else:
            config_cls = TransformersBitsAndBytesConfig if component == "text_encoder" else DiffusersBitsAndBytesConfig
            quantization_config = {component: create_quant_config(config_cls, skip_modules)}

        config_info = {name: quant_config_to_info(config) for name, config in quantization_config.items()}

        return {
            "quantization_config": quantization_config,
            "config_info": config_info,
        }


class AutoModelLoader(NodeBase):
    label = "Load Model"
    category = "loader"
    resizable = True
    skipParamsCheck = True
    params = {
        "model_type": {
            "label": "Model Type",
            "type": "string",
            "options": {
                "": "",
                "unet": "UNet",
                "transformer": "Transformer",
                "vae": "VAE",
                "controlnet": "ControlNet",
            },
            "onChange": [
                "set_filters",
                {"action": "signal", "target": "model"},
            ],
        },
        "model_id": {
            "label": "Model ID",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": [""]},
                    "local": {"className": [""]},
                },
            },
        },
        "dtype": {
            "label": "dtype",
            "options": ["float32", "float16", "bfloat16"],
            "value": "float16",
            "postProcess": str_to_dtype,
        },
        "subfolder": {"label": "Subfolder", "type": "string", "value": ""},
        "variant": {"type": "string", "value": "", "options": ["", "fp16", "bf16"]},
        "trust_remote_code": {"label": "Trust Remote Code", "type": "boolean", "value": False},
        "revision": {
            "label": "Revision",
            "type": "string",
            "value": "",
            "description": "Required 40-character commit hash when Trust Remote Code is enabled.",
        },
        "device": {"label": "Device", "type": "string", "value": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "auto_offload": {"label": "Enable Auto Offload", "type": "boolean", "value": True},
        "offload_mode": offload_mode_param(),
        "model": {"label": "Model", "display": "output", "type": "diffusers_auto_model"},
    }

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
        super().__del__()

    def set_filters(self, values, ref):
        model_type = values.get("model_type", "")

        filters = []

        if model_type == "unet":
            filters = ["UNet2DConditionModel"]
            self.set_field_params("subfolder", {"value": "unet"})
        elif model_type == "transformer":
            filters = ["QwenImageTransformer2DModel", "FluxTransformer2DModel", "SD3Transformer2DModel"]
            self.set_field_params("subfolder", {"value": "transformer"})
        elif model_type == "vae":
            filters = ["AutoencoderKL", "AutoencoderKLQwenImage"]
            self.set_field_params("subfolder", {"value": "vae"})
        elif model_type == "controlnet":
            filters = ["ControlNetModel", "QwenImageControlNetModel", "FluxControlNetModel"]
            self.set_field_params("subfolder", {"value": ""})

        self.set_field_params(
            "model_id",
            {
                "fieldOptions": {
                    "filter": {
                        "hub": {"className": filters},
                    },
                },
            },
        )

    def execute(
        self,
        model_type,
        model_id,
        dtype,
        trust_remote_code,
        device=DEFAULT_DEVICE,
        auto_offload=True,
        offload_mode=OFFLOAD_MODE_MODEL_CPU,
        variant=None,
        subfolder=None,
        revision=None,
    ):
        logger.debug(f"AutoModelLoader ({self.node_id}) received parameters:")
        logger.debug(f"  model_type: '{model_type}'")
        logger.debug(f"  model_id: '{model_id}'")
        logger.debug(f"  subfolder: '{subfolder}'")
        logger.debug(f"  variant: '{variant}'")
        logger.debug(f"  trust_remote_code: '{trust_remote_code}'")
        logger.debug(f"  dtype: '{dtype}'")
        logger.debug(f"  device: '{device}'")
        logger.debug(f"  auto_offload: '{auto_offload}'")
        logger.debug(f"  offload_mode: '{offload_mode}'")

        supported_model_types = {"unet", "transformer", "vae", "controlnet"}
        if model_type not in supported_model_types:
            raise ValueError(
                "AutoModelLoader requires a component type of unet, transformer, vae, or controlnet; "
                f"received {model_type!r}. Rebuild or repair the managed graph before loading model weights."
            )

        if isinstance(model_id, dict):
            real_model_id = model_id.get("value", model_id)
            _source = model_id.get("source", "hub")
        else:
            real_model_id = ""

        if real_model_id == "":
            self.notify(
                "Please provide a valid Repository ID.",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        revision = resolve_model_revision(real_model_id, revision, source=_source)
        revision = require_immutable_hub_revision(
            real_model_id,
            revision,
            required=bool(trust_remote_code),
        )

        # Normalize parameters
        variant = None if variant == "" else variant
        subfolder = None if subfolder == "" else subfolder

        normalized_offload_mode = normalize_offload_mode(
            offload_mode,
            auto_offload=bool(auto_offload),
            device=device,
        )
        spec = ComponentSpec(
            name=model_type,
            repo=real_model_id,
            subfolder=subfolder,
            variant=variant,
            revision=revision,
        )
        reusable = reusable_standalone_component(
            components,
            name=model_type,
            load_id=spec.load_id,
            dtype=dtype,
            offload_mode=normalized_offload_mode,
            device=device,
            node_id=self.node_id,
        )
        if reusable:
            _existing_id, model = reusable
            self.progress(
                99,
                phase="loading",
                message=f"Reusing resident {model_type} from {real_model_id}",
            )
            offload_result = None
        else:
            self.progress(
                0,
                phase="loading",
                message=f"Loading {model_type} weights from {real_model_id}",
            )
            with self.diffusers_loading_progress():
                model = spec.load(torch_dtype=dtype, trust_remote_code=trust_remote_code)
            self.progress(
                99,
                phase="component_placement",
                message=f"Placing {model_type} on {device}; this one-time accelerator copy can take several minutes",
            )
            offload_result = apply_model_offload(
                model,
                component_name=model_type,
                mode=normalized_offload_mode,
                device=device,
                node_id=self.node_id,
                scope="modular-auto-model",
            )
            model._modiff_offload_mode = normalized_offload_mode
            model._modiff_execution_device = str(torch.device(device))
            model._modiff_offload_node_id = (
                str(self.node_id) if normalized_offload_mode == OFFLOAD_MODE_GROUP_DISK else None
            )
        logger.debug(
            " AutoModelLoader: applied %s via %s to %s",
            offload_result.mode if offload_result else normalized_offload_mode,
            offload_result.method if offload_result else "resident_reuse",
            offload_result.components if offload_result else [model_type],
        )
        comp_id = components.add(model_type, model, collection=self.node_id)
        logger.debug(f" AutoModelLoader: comp_id added: {comp_id}")
        logger.debug(f" AutoModelLoader: component manager: {components}")

        model = components.get_model_info(comp_id)
        model["repo_id"] = real_model_id
        model["revision"] = revision
        model["trust_remote_code"] = bool(trust_remote_code)

        return {"model": model}


class ModelsLoader(NodeBase):
    label = "Load Models"
    category = "loader"
    resizable = True
    skipParamsCheck = True
    params = {
        "model_type": {
            "label": "Model Type",
            "type": "string",
            "options": {
                "": "",
            },
            "onChange": [
                "set_filters",
                {"action": "signal", "target": "unet_out"},
                {"action": "signal", "target": "text_encoders"},
                {"action": "signal", "target": "vae_out"},
                {"action": "signal", "target": "image_encoder"},
            ],
        },
        "repo_id": {
            "label": "Repository ID",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ""},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": [""]},
                    "local": {"className": [""]},
                },
            },
        },
        "dtype": {
            "label": "dtype",
            "options": ["float32", "float16", "bfloat16"],
            "value": "float16",
            "postProcess": str_to_dtype,
        },
        "device": {"label": "Device", "type": "string", "value": DEFAULT_DEVICE, "options": DEVICE_LIST},
        "trust_remote_code": {"label": "Trust Remote Code", "type": "boolean", "value": False},
        "revision": {
            "label": "Revision",
            "type": "string",
            "value": "",
            "description": "Required 40-character commit hash for custom or trusted remote code.",
        },
        "auto_offload": {"label": "Enable Auto Offload", "type": "boolean", "value": True},
        "offload_mode": offload_mode_param(
            modes=[OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK]
        ),
        "unet": {"label": "Denoise Model", "display": "input", "type": "diffusers_auto_model"},
        "vae": {"label": "VAE", "display": "input", "type": "diffusers_auto_model"},
        "lora_list": {"label": "Lora", "display": "input", "type": "custom_lora"},
        "text_encoders": {"label": "Text Encoders", "display": "output", "type": "diffusers_auto_models"},
        "unet_out": {"label": "Denoise Model", "display": "output", "type": "diffusers_auto_model"},
        "vae_out": {"label": "VAE", "display": "output", "type": "diffusers_auto_model"},
        "scheduler": {"label": "Scheduler", "display": "output", "type": "diffusers_auto_model"},
        "image_encoder": {"label": "Image Encoder", "display": "output", "type": "diffusers_auto_model"},
        "quant_config": {"label": "Quant Config", "display": "input", "type": "quant_config"},
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self.loader = None
        self.model_types_loaded = False

    def __del__(self):
        node_comp_ids = components._lookup_ids(collection=self.node_id)
        for comp_id in node_comp_ids:
            components.remove_from_collection(comp_id, self.node_id)
        self.loader = None
        super().__del__()

    def set_filters(self, values, ref):
        # first time dynamically load the model_type options
        if not self.model_types_loaded:
            self.set_field_params("model_type", {"options": get_all_model_types()})
            self.model_types_loaded = True

        model_type = values.get("model_type", "")
        metadata = get_model_type_metadata(model_type)

        if metadata:
            default_dtype = metadata["default_dtype"]
        else:
            # Fallback for empty or unknown model types
            default_dtype = "float16"
        filters = [model_type]  # Model types map one-to-one to modular pipeline classes.

        self.set_field_params(
            "repo_id",
            {
                "fieldOptions": {
                    "filter": {"hub": {"className": filters}},
                },
            },
        )
        self.set_field_params("dtype", {"value": default_dtype})

    def execute(
        self,
        model_type,
        repo_id,
        device,
        dtype,
        unet=None,
        vae=None,
        lora_list=None,
        trust_remote_code=False,
        auto_offload=True,
        offload_mode=OFFLOAD_MODE_MODEL_CPU,
        quant_config=None,
        revision=None,
    ):
        requested_offload_mode = offload_mode
        offload_mode = normalize_offload_mode(
            offload_mode,
            auto_offload=auto_offload,
            device=device,
        )
        self._loader_diagnostics = {
            "node_id": self.node_id,
            "loader": "ModelsLoader",
            "model_type": model_type,
            "repo_id": None,
            "dtype": str(dtype),
            "graph_offload_mode": safe_diagnostic_value(requested_offload_mode),
            "normalized_offload_mode": offload_mode,
            "auto_offload": bool(auto_offload),
            "trust_remote_code": bool(trust_remote_code),
            "quantized_components": [],
            "components_to_load": [],
            "components_reused": [],
            "components_loaded": [],
            "components_failed": [],
            "components_skipped": [],
            "components_unknown": [],
            "offload": {
                "mode": offload_mode,
                "incremental": False,
                "components": [],
            },
        }
        if offload_mode not in [
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ]:
            self.notify(
                f"Modular Diffusers ModelsLoader does not support {offload_mode} offload.",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        try:
            quant_config = normalize_quant_config_input(quant_config)
        except TypeError as exc:
            self.notify(str(exc), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            raise exc
        self._loader_diagnostics["quantized_components"] = sorted(list(quant_config.keys())) if quant_config else []
        self._loader_diagnostics["quantization"] = safe_diagnostic_value(component_quant_config_summary(quant_config))

        logger.debug(f"""
            ModelsLoader ({self.node_id}) received parameters:
            - repo_id: {repo_id}
            - dtype: {dtype}
            - device: {device}
            - unet: {unet}
            - vae: {vae}
            - quant_config: {quant_config}
            - auto_offload: {auto_offload}
            - offload_mode: {offload_mode}
            - trust_remote_code: {trust_remote_code}
            - lora_list: {lora_list}
            - model_type: {model_type}
        """)

        components_to_update = {}

        if unet:
            components_to_update.update(
                components.get_components_by_ids(ids=[unet["model_id"]], return_dict_with_names=True)
            )
        if vae:
            components_to_update.update(
                components.get_components_by_ids(ids=[vae["model_id"]], return_dict_with_names=True)
            )

        if isinstance(repo_id, dict):
            real_repo_id = repo_id.get("value", repo_id)
            _source = repo_id.get("source", "hub")
        else:
            real_repo_id = ""

        if real_repo_id == "":
            self.notify(
                "Please provide a valid Repository ID.",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None
        self._loader_diagnostics["repo_id"] = real_repo_id
        revision = resolve_model_revision(
            real_repo_id,
            revision,
            model_type=model_type,
            source=_source,
        )
        revision = require_immutable_hub_revision(
            real_repo_id,
            revision,
            required=bool(trust_remote_code) or model_type == "DummyCustomPipeline",
        )
        self._loader_diagnostics["revision"] = revision

        use_group_offload = offload_mode in [OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK]

        configure_components_manager_offload(components, mode=offload_mode, device=device)

        self.loader = ModularPipeline.from_pretrained(
            real_repo_id,
            components_manager=components,
            collection=self.node_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=True,
        )
        self._loader_diagnostics["component_revision_pins"] = pin_modular_component_revisions(
            self.loader,
            real_repo_id,
            revision,
        )

        if model_type == "DummyCustomPipeline":
            # update node param
            custom_config = PipelineConfig.load(real_repo_id, revision=revision, local_files_only=True)
            custom_config.label = "Custom"

            # update repo_id for DummyCustomPipeline
            DummyCustomPipeline.repo_id = real_repo_id
            DummyCustomPipeline.revision = revision
            DummyCustomPipeline.trust_remote_code = bool(trust_remote_code)
            # register DummyCustomPipeline to MODULAR_REGISTRY
            MODULAR_REGISTRY.register(DummyCustomPipeline, custom_config)

        else:
            DummyCustomPipeline.repo_id = None
            DummyCustomPipeline.revision = None
            DummyCustomPipeline.trust_remote_code = False
            MODULAR_REGISTRY.register(DummyCustomPipeline, DUMMY_CUSTOM_PIPELINE_CONFIG)

        ALL_COMPONENTS = self.loader.pretrained_component_names

        text_node = self.loader.blocks.sub_blocks["text_encoder"].init_pipeline(real_repo_id)
        text_encoder_names = text_node.pretrained_component_names

        components_to_load = [c for c in ALL_COMPONENTS if c not in components_to_update]
        components_to_reload = []

        denoiser_options = ["unet", "transformer"]
        denoiser_name = next((option for option in denoiser_options if option in ALL_COMPONENTS), None)

        for comp_name in components_to_load:
            comp_spec = self.loader.get_component_spec(comp_name)
            if comp_spec.load_id != "null":
                # Reuse requires the same model identity, dtype, quantization,
                # execution device, and offload policy. In particular, never
                # carry a group-offload hook into a differently configured run.
                comp_ids_to_reuse = reusable_component_ids(
                    components,
                    name=comp_name,
                    load_id=comp_spec.load_id,
                    dtype=dtype,
                    requested_quantization=quant_config.get(comp_name) if quant_config else None,
                    offload_mode=offload_mode,
                    device=device,
                    node_id=self.node_id,
                )

                if not comp_ids_to_reuse:
                    components_to_reload.append(comp_name)
                else:
                    # Reuse existing component
                    self._loader_diagnostics.setdefault("components_reused", []).append(comp_name)
                    components_to_update.update(
                        components.get_components_by_ids(ids=comp_ids_to_reuse, return_dict_with_names=True)
                    )

        required_components = {denoiser_name, "vae", "scheduler", *text_encoder_names}
        if model_type == "WanImage2VideoModularPipeline":
            required_components.add("image_encoder")
        required_components = {name for name in required_components if name}
        self._loader_diagnostics["required_components"] = sorted(required_components)
        self._loader_diagnostics["components_to_load"] = list(components_to_reload)

        incremental_group_offload = should_incrementally_group_offload(
            use_group_offload=use_group_offload,
            quant_config=quant_config,
        )
        with self.diffusers_loading_progress():
            load_components_strict(
                self.loader,
                names=components_to_reload,
                required_names=required_components,
                model_id=real_repo_id,
                dtype=dtype,
                offload_mode=offload_mode,
                quant_config=quant_config,
                diagnostics=self._loader_diagnostics,
                component_load_kwargs={
                    "torch_dtype": dtype,
                    "trust_remote_code": trust_remote_code,
                    "quantization_config": quant_config,
                },
                incremental_group_offload={
                    "device": device,
                    "mode": offload_mode,
                    "node_id": self.node_id,
                    "scope": "modular-diffusers",
                }
                if incremental_group_offload
                else None,
            )
        self.loader.update_components(**components_to_update)

        if use_group_offload:
            try:
                offload_result = apply_component_group_offload(
                    self.loader,
                    component_names=DEFAULT_GROUP_COMPONENTS,
                    device=device,
                    mode=offload_mode,
                    node_id=self.node_id,
                    scope="modular-diffusers",
                )
                if not offload_result.applied:
                    raise RuntimeError("No compatible Modular Diffusers component was available to offload.")
                self._loader_diagnostics["offload"].update(
                    {
                        "mode": offload_result.mode,
                        "method": offload_result.method,
                        "components": sorted(
                            set(self._loader_diagnostics["offload"].get("components", []) + offload_result.components)
                        ),
                        "disk_path": offload_result.disk_path,
                        "detail": offload_result.detail,
                    }
                )
                logger.debug(f" ModelsLoader: applied {offload_mode} to {offload_result.components}")
            except RuntimeError as exc:
                self.notify(str(exc), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
                raise
        elif offload_mode == "none":
            resident_modules = place_pipeline_components(
                self.loader,
                device,
                progress_callback=lambda name, index, total: self.progress(
                    99,
                    phase="component_placement",
                    message=(
                        f"Placing {name} on {device} ({index}/{total}); "
                        "this one-time accelerator copy can take several minutes"
                    ),
                    current_step=index,
                    total_steps=total,
                ),
            )
            self._loader_diagnostics["offload"].update(
                {
                    "mode": offload_mode,
                    "method": "to_device",
                    "components": resident_modules,
                }
            )
        elif offload_mode == OFFLOAD_MODE_MODEL_CPU:
            self._loader_diagnostics["offload"].update(
                {
                    "mode": offload_mode,
                    "method": "components_manager_auto_cpu_offload",
                    "components": [],
                }
            )

        record_pipeline_component_runtime_policy(
            self.loader,
            offload_mode=offload_mode,
            device=device,
            node_id=self.node_id,
        )

        print(f" ModelsLoader: reloaded components: {components_to_reload}")
        print(f" ModelsLoader: updated components: {components_to_update.keys()}")

        if hasattr(self.loader, "unload_lora_weights"):
            self.loader.unload_lora_weights()
        if lora_list:
            update_lora_adapters(self.loader, lora_list)
            apply_lora_scheduler_override(self.loader, lora_list)

        # Construct loaded_components at the end after all modifications
        try:
            loaded_components = {
                "unet_out": node_get_component_info(node_id=self.node_id, manager=components, name=denoiser_name),
                "vae_out": node_get_component_info(node_id=self.node_id, manager=components, name="vae"),
                "text_encoders": {
                    k: node_get_component_info(node_id=self.node_id, manager=components, name=k)
                    for k in text_encoder_names
                },
                "scheduler": node_get_component_info(node_id=self.node_id, manager=components, name="scheduler"),
            }

            if model_type == "WanImage2VideoModularPipeline":
                loaded_components["image_encoder"] = node_get_component_info(
                    node_id=self.node_id, manager=components, name="image_encoder"
                )
        except ValueError as e:
            self.notify(
                f" ModelsLoader: Error retrieving component info: {e}",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            self._loader_diagnostics.setdefault("components_failed", []).append(
                {
                    "name": "component_info",
                    "required": True,
                    "error": str(e),
                }
            )
            raise RuntimeError(f"ModelsLoader could not retrieve required component info: {e}") from e

        # Make every connected output self-describing. Runtime cleanup may
        # recreate downstream nodes without replaying their dynamic UI signal.
        for k, v in loaded_components.items():
            if isinstance(v, dict):
                v["repo_id"] = real_repo_id
                v["model_type"] = model_type
                v["revision"] = revision
                v["trust_remote_code"] = bool(trust_remote_code)

        logger.debug(f" ModelsLoader: Final component_manager state: {components}")

        return loaded_components
