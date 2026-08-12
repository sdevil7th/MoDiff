# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
from collections.abc import Mapping

from diffusers import LayerSkipConfig, SmoothedEnergyGuidanceConfig, guiders as diffusers_guiders

from modiff.NodeBase import NodeBase

from . import MODULAR_GUIDER_OPTIONS, MODULAR_LAYER_BLOCK_OPTIONS
from .pipeline_schema import MAX_GUIDER_OPTIONS, MAX_LAYER_BLOCK_OPTIONS


logger = logging.getLogger("modiff")

LAYER_CONFIG_MAPPING = {
    "SkipLayerGuidance": "skip_layer_config",
    "AutoGuidance": "auto_guidance_config",
    "SmoothedEnergyGuidance": "seg_guidance_config",
    "PerturbedAttentionGuidance": "perturbed_guidance_config",
}

GUIDER_OPTIONS = {
    "ClassifierFreeGuidance": "Classifier Free Guidance",
    "SkipLayerGuidance": "Skip Layer Guidance",
    "AdaptiveProjectedGuidance": "Adaptive Projected Guidance",
    "AdaptiveProjectedMixGuidance": "Adaptive Projected Mix Guidance",
    "MagnitudeAwareGuidance": "Magnitude Aware Guidance",
    "ClassifierFreeZeroStarGuidance": "Classifier Free Zero Star Guidance",
    "AutoGuidance": "Auto Guidance",
    "SmoothedEnergyGuidance": "Smoothed Energy Guidance",
    "PerturbedAttentionGuidance": "Perturbed Attention Guidance",
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
    "AdaptiveProjectedMixGuidance": {
        "adaptive_projected_guidance_scale": {
            "label": "Adaptive Projected Guidance Scale",
            "type": "float",
            "value": 10.0,
            "min": 0.0,
            "max": 100.0,
        },
        "adaptive_projected_guidance_momentum": {
            "label": "Adaptive Projected Guidance Momentum",
            "type": "float",
            "value": -0.5,
            "min": -1.0,
            "max": 1.0,
            "step": 0.01,
        },
        "adaptive_projected_guidance_rescale": {
            "label": "Adaptive Projected Guidance Rescale",
            "type": "float",
            "value": 10.0,
            "min": 0.0,
            "max": 100.0,
        },
        "eta": {
            "label": "Eta",
            "type": "float",
            "value": 0.0,
            "step": 0.01,
        },
        "adaptive_projected_guidance_start_step": {
            "label": "Adaptive Projected Guidance Start Step",
            "type": "int",
            "value": 5,
            "min": 0,
        },
    },
    "MagnitudeAwareGuidance": {
        "alpha": {
            "label": "Magnitude Suppression Alpha",
            "type": "float",
            "value": 8.0,
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
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
    "PerturbedAttentionGuidance": {
        "perturbed_guidance_scale": {
            "label": "Perturbed Guidance Scale",
            "type": "float",
            "value": 2.8,
            "min": 0.0,
            "max": 10.0,
        },
        "perturbed_guidance_start": {
            "label": "Perturbed Guidance Start",
            "type": "float",
            "display": "slider",
            "value": 0.01,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        "perturbed_guidance_stop": {
            "label": "Perturbed Guidance Stop",
            "type": "float",
            "display": "slider",
            "value": 0.2,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
    },
}

_BOOLEAN_GUIDER_ARGUMENTS = frozenset({"enabled", "use_original_formulation"})
_INTEGER_GUIDER_ARGUMENTS = frozenset({"adaptive_projected_guidance_start_step", "zero_init_steps"})


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
                    "PerturbedAttentionGuidance": ["layers_config"],
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
        "enabled": {
            "label": "Enabled",
            "type": "boolean",
            "value": True,
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
            "onSignal": [
                {
                    "action": "value",
                    "target": "guider",
                    "prop": "options",
                    "data": MODULAR_GUIDER_OPTIONS,
                },
                {"action": "signal", "target": "layers_config"},
            ],
        },
        "layers_config": {"label": "Layers", "type": "layers_config", "display": "input"},
    }

    def _selected_guider(self, guider):
        model_type = self.get_signal_value("guider_out")
        allowed = MODULAR_GUIDER_OPTIONS.get(model_type) if isinstance(model_type, str) else None
        if (
            not isinstance(guider, str)
            or not isinstance(allowed, list)
            or len(allowed) > MAX_GUIDER_OPTIONS
            or guider not in GUIDER_OPTIONS
            or guider not in allowed
        ):
            raise ValueError("Guider requires a class allowed by the connected reviewed Modular pipeline.")
        return guider

    def updateNode(self, values, ref):
        value = self._selected_guider(values.get("guider"))

        params = GUIDER_CONFIGS.get(value, {})
        self.send_node_definition(params)

    def execute(self, guider, layers_config=None, **kwargs):
        logger.debug(f" Guider ({self.node_id}) received parameters:")
        logger.debug(f" - guider: {guider}")
        logger.debug(f" - kwargs: {kwargs}")

        guider_options = {}

        for key, value in kwargs.items():
            if key in _BOOLEAN_GUIDER_ARGUMENTS:
                if not isinstance(value, bool):
                    raise TypeError(f"{key} must be a boolean.")
                guider_options[key] = value
            elif key in _INTEGER_GUIDER_ARGUMENTS:
                numeric_value = float(value)
                if isinstance(value, bool) or not numeric_value.is_integer():
                    raise ValueError(f"{key} must be an integer.")
                guider_options[key] = int(numeric_value)
            else:
                guider_options[key] = float(value)

        logger.debug(f" - guider options: {guider_options}")

        guider = self._selected_guider(guider)

        guider_cls = getattr(diffusers_guiders, guider)

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
            elif isinstance(layers_config, (LayerSkipConfig, SmoothedEnergyGuidanceConfig)):
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
                        indices = getattr(config_dict, "indices", None)
                        fqn = getattr(config_dict, "fqn", None)
                        if (
                            not isinstance(indices, list)
                            or not indices
                            or any(
                                isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices
                            )
                        ):
                            raise ValueError(
                                f"{guider} layer indices must be a non-empty list of non-negative integers."
                            )
                        if not isinstance(fqn, str) or not fqn or fqn != fqn.strip():
                            raise ValueError(
                                f"{guider} requires a non-empty layer FQN without surrounding whitespace."
                            )
                        if guider == "PerturbedAttentionGuidance":
                            layer_config_values = config_dict.to_dict()
                            if float(layer_config_values.get("dropout", 1.0)) != 1.0:
                                raise ValueError(
                                    "PerturbedAttentionGuidance requires Layers dropout to be 1.0 because it "
                                    "perturbs attention scores."
                                )
                            layer_config_values.update(
                                skip_attention=False,
                                skip_attention_scores=True,
                                skip_ff=False,
                            )
                            config_dict = LayerSkipConfig(**layer_config_values)
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
                        layer_config_values = dict(config_dict)
                        if guider == "PerturbedAttentionGuidance":
                            if float(layer_config_values.get("dropout", 1.0)) != 1.0:
                                raise ValueError(
                                    "PerturbedAttentionGuidance requires Layers dropout to be 1.0 because it "
                                    "perturbs attention scores."
                                )
                            layer_config_values.update(
                                skip_attention=False,
                                skip_attention_scores=True,
                                skip_ff=False,
                            )
                        layer_config = LayerSkipConfig(**layer_config_values)

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
                "data": MODULAR_LAYER_BLOCK_OPTIONS,
            },
        },
    }

    def _selected_blocks(self, values):
        if not isinstance(values, Mapping):
            raise TypeError("Layers values must be a mapping.")
        blocks_select = values.get("blocks_select", [])
        if (
            not isinstance(blocks_select, list)
            or len(blocks_select) > MAX_LAYER_BLOCK_OPTIONS
            or any(not isinstance(block, str) or not block or block != block.strip() for block in blocks_select)
            or len(blocks_select) != len(set(blocks_select))
        ):
            raise ValueError("Layers requires a bounded list of unique block names.")
        if not blocks_select:
            return ()

        model_type = self.get_signal_value("layers_config")
        allowed_blocks = MODULAR_LAYER_BLOCK_OPTIONS.get(model_type) if isinstance(model_type, str) else None
        if not isinstance(allowed_blocks, list) or any(block not in allowed_blocks for block in blocks_select):
            raise ValueError("Layers requires block names allowed by the connected reviewed Modular pipeline.")
        return tuple(blocks_select)

    def set_blocks(self, values, ref):
        blocks_select = self._selected_blocks(values)

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
        blocks_select = self._selected_blocks(kwargs)
        supplied_blocks = {block for block in kwargs if block != "blocks_select"}
        if supplied_blocks != set(blocks_select):
            raise ValueError("Layers inputs must exactly match the reviewed selected block names.")

        for block in blocks_select:
            config = kwargs.get(block, {})
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
