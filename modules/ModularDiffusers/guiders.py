# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging

from diffusers import LayerSkipConfig, SmoothedEnergyGuidanceConfig

from modiff.NodeBase import NodeBase

from . import FLUX_BLOCKS, QWEN_IMAGE_BLOCKS, SDXL_BLOCKS


logger = logging.getLogger("modiff")

LAYER_CONFIG_MAPPING = {
    "SkipLayerGuidance": "skip_layer_config",
    "AutoGuidance": "auto_guidance_config",
    "SmoothedEnergyGuidance": "seg_guidance_config",
}

GUIDER_OPTIONS = {
    "ClassifierFreeGuidance": "Classifier Free Guidance",
    "SkipLayerGuidance": "Skip Layer Guidance",
    "AdaptiveProjectedGuidance": "Adaptive Projected Guidance",
    "ClassifierFreeZeroStarGuidance": "Classifier Free Zero Star Guidance",
    "AutoGuidance": "Auto Guidance",
    "SmoothedEnergyGuidance": "Smoothed Energy Guidance",
    "TangentialClassifierFreeGuidance": "Tangential Classifier Free Guidance",
    "FrequencyDecoupledGuidance": "Frequency Decoupled Guidance",
}

GUIDER_CONFIGS = {
    "SkipLayerGuidance": {
        "skip_layer_guidance_scale": {
            "label": "Skip Layer Guidance Scale",
            "type": "float",
            "value": 2.8,
            "min": 0.0,
            "max": 10.0,
        },
        "skip_layer_guidance_start": {
            "label": "Skip Layer Start",
            "type": "float",
            "display": "slider",
            "default": 0.01,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        "skip_layer_guidance_stop": {
            "label": "Stop",
            "type": "float",
            "display": "slider",
            "default": 0.2,
            "min": 0.02,
            "max": 1.0,
            "step": 0.01,
        },
    },
    "AdaptiveProjectedGuidance": {
        "adaptive_projected_guidance_momentum": {
            "label": "Adaptive Projected Guidance Momentum",
            "type": "float",
            "value": 0.0,
            "min": -1.0,
            "max": 1.0,
            "step": 0.01,
        },
        "adaptive_projected_guidance_rescale": {
            "label": "Adaptive Projected Guidance Rescale",
            "type": "float",
            "value": 15.0,
            "min": 0.0,
            "max": 100.0,
        },
    },
    "ClassifierFreeZeroStarGuidance": {
        "zero_init_steps": {
            "label": "Zero Init Steps",
            "type": "int",
            "value": 1,
        },
    },
    "AutoGuidance": {
        "dropout": {
            "label": "Dropout",
            "type": "float",
            "value": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        }
    },
}


class Guider(NodeBase):
    label = "Guider"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    params = {
        "guider": {
            "label": "Guider",
            "fieldOptions": {"loading": True},
            "type": "string",
            "options": GUIDER_OPTIONS,
            "value": "ClassifierFreeGuidance",
            "onChange": [
                "updateNode",
                {
                    "SkipLayerGuidance": ["layers_config"],
                    "AutoGuidance": ["layers_config"],
                    "SmoothedEnergyGuidance": ["layers_config"],
                },
            ],
        },
        "guidance_scale": {
            "label": "Guidance Scale",
            "type": "float",
            "display": "slider",
            "value": 5,
            "min": 1.0,
            "max": 20.0,
            "step": 0.1,
        },
        "guidance_rescale": {
            "label": "Guidance Rescale",
            "type": "float",
            "display": "slider",
            "value": 0.0,
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
        },
        "use_original_formulation": {
            "label": "Original Formulation",
            "type": "boolean",
            "value": False,
        },
        "start": {
            "label": "Start",
            "type": "float",
            "display": "slider",
            "value": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        "stop": {
            "label": "Stop",
            "type": "float",
            "display": "slider",
            "value": 1.0,
            "min": 0.01,
            "max": 1.0,
            "step": 0.01,
        },
        "guider_out": {
            "label": "Guider",
            "display": "output",
            "type": "custom_guider",
            "onSignal": {
                "action": "signal",
                "target": "layers_config",
            },
        },
        "layers_config": {"label": "Layers", "type": "layers_config", "display": "input"},
    }

    def updateNode(self, values, ref):
        value = values.get("guider")

        params = GUIDER_CONFIGS.get(value, {})
        self.send_node_definition(params)

    def execute(self, guider, layers_config=None, **kwargs):
        logger.debug(f" Guider ({self.node_id}) received parameters:")
        logger.debug(f" - guider: {guider}")
        logger.debug(f" - kwargs: {kwargs}")

        guider_options = {}

        for key, value in kwargs.items():
            if key == "use_original_formulation":
                guider_options[key] = value
            else:
                guider_options[key] = float(value)

        logger.debug(f" - guider options: {guider_options}")

        if guider not in GUIDER_OPTIONS:
            raise ValueError(f"Unsupported Diffusers guider: {guider!r}.")

        guider_cls = getattr(__import__("diffusers", fromlist=[guider]), guider)

        configs = {}

        if guider in LAYER_CONFIG_MAPPING:
            if layers_config is None or layers_config == []:
                raise ValueError(
                    f"{guider} requires a non-empty Layers connection. "
                    "Select a transformer-block stack and at least one layer index."
                )
            config_arg_name = LAYER_CONFIG_MAPPING[guider]

            if isinstance(layers_config, dict):
                layers_config = [layers_config]

            if isinstance(layers_config, list):
                layer_configs = []

                for config_dict in layers_config:
                    if not isinstance(config_dict, dict):
                        expected_type = (
                            SmoothedEnergyGuidanceConfig if guider == "SmoothedEnergyGuidance" else LayerSkipConfig
                        )
                        if not isinstance(config_dict, expected_type):
                            raise TypeError(
                                f"{guider} layer entries must be mappings or {expected_type.__name__} instances."
                            )
                        layer_configs.append(config_dict)
                        continue

                    indices = config_dict.get("indices")
                    fqn = config_dict.get("fqn")
                    if (
                        not isinstance(indices, list)
                        or not indices
                        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
                    ):
                        raise ValueError(f"{guider} layer indices must be a non-empty list of non-negative integers.")
                    if not isinstance(fqn, str) or not fqn or fqn != fqn.strip():
                        raise ValueError(f"{guider} requires a non-empty layer FQN without surrounding whitespace.")

                    if guider == "SmoothedEnergyGuidance":
                        layer_config = SmoothedEnergyGuidanceConfig(indices=indices, fqn=fqn)
                    else:
                        layer_config = LayerSkipConfig(**config_dict)

                    layer_configs.append(layer_config)

                configs[guider] = {config_arg_name: layer_configs}
            else:
                expected_type = SmoothedEnergyGuidanceConfig if guider == "SmoothedEnergyGuidance" else LayerSkipConfig
                if not isinstance(layers_config, expected_type):
                    raise TypeError(
                        f"{guider} Layers input must be a mapping, list, or {expected_type.__name__} instance."
                    )
                configs[guider] = {config_arg_name: layers_config}

        options = {**guider_options}

        if guider == "FrequencyDecoupledGuidance" and "guidance_scale" in options:
            # The upstream constructor intentionally accepts one scale per
            # frequency level under the plural ``guidance_scales`` name. Keep
            # MoDiff's existing single-value control as a one-level list.
            options["guidance_scales"] = [options.pop("guidance_scale")]

        if guider in configs:
            options.update(configs[guider])

        guider_out = guider_cls(**options)

        return {"guider_out": guider_out}


class Layers(NodeBase):
    label = "Layers"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    params = {
        "blocks_select": {
            "label": "Blocks",
            "type": "string",
            "display": "select",
            "options": {"": ""},
            "fieldOptions": {"multiple": True},
            "onChange": "set_blocks",
        },
        "layers_config": {
            "label": "Layers",
            "display": "output",
            "type": "layers_config",
            "onSignal": {
                "action": "value",
                "target": "blocks_select",
                "prop": "options",
                "data": {
                    "StableDiffusionXLModularPipeline": SDXL_BLOCKS,
                    "QwenImageModularPipeline": QWEN_IMAGE_BLOCKS,
                    "QwenImageEditModularPipeline": QWEN_IMAGE_BLOCKS,
                    "QwenImageEditPlusModularPipeline": QWEN_IMAGE_BLOCKS,
                    "FluxModularPipeline": FLUX_BLOCKS,
                    "FluxKontextModularPipeline": FLUX_BLOCKS,
                },
            },
        },
    }

    def set_blocks(self, values, ref):
        blocks_select = values.get("blocks_select", [])

        params = {}

        for block_name in blocks_select:
            params[block_name] = {
                "label": block_name,
                "display": "layerconfig",
                "value": {"enabled": True, "indices": "", "dropout_visible": True, "skip_checkboxes_visible": True},
            }
        self.send_node_definition(params)

    def execute(self, **kwargs):
        layer_configs = []

        for block in kwargs:
            if block == "blocks_select":
                continue

            config = kwargs.get(block, {})

            if not isinstance(block, str) or not block or block != block.strip():
                raise ValueError("Layer block names must be non-empty FQNs without surrounding whitespace.")
            if not isinstance(config, dict):
                raise TypeError(f"Layer configuration for {block!r} must be a mapping.")

            indices_str = str(config.get("indices", ""))
            try:
                indices = [int(value.strip()) for value in indices_str.split(",") if value.strip()]
            except ValueError as error:
                raise ValueError(f"Layer indices for {block!r} must be comma-separated integers.") from error
            if not indices:
                indices = [0]
            if any(index < 0 for index in indices):
                raise ValueError(f"Layer indices for {block!r} must be non-negative.")

            layer_config = {
                "indices": indices,
                "fqn": block,
                "dropout": config.get("dropout", 1.0),
                "skip_attention": config.get("skip_attention", False),
                "skip_attention_scores": config.get("skip_attention_scores", False),
                "skip_ff": config.get("skip_ff", False),
            }

            layer_configs.append(layer_config)

        return {"layers_config": layer_configs}
