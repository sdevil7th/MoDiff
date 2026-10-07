"""Project an existing loader bundle onto the three reviewed native inputs.

No pipeline, model, owner or ambient component is created here. The original
bundle keeps cache identity; temporary role projections carry its exact token.
"""

from modiff.component_bundle_contracts import QWEN_T2I_BUNDLE_INPUTS, QWEN_T2I_BUNDLE_PIPELINE

from .route_state import ROUTE_STATE_INPUT, bind_loader_outputs, require_component_binding, resolve_managed_component_by_id
from .utils import collect_model_ids


_OUTPUTS = {"text_encoders": ("text_encoders", "text_encoders"), "unet": ("unet_out", "denoiser"),
            "scheduler": ("scheduler", "scheduler"), "vae": ("vae_out", "vae")}
_IDENTITY_FIELDS = ("model_type", "repo_id", "repo_source", "revision", "trust_remote_code")


def normalize_component_bundle_inputs(kwargs, *, node_type, component_manager):
    """Validate demanded members before native initialization or cache reuse."""
    if kwargs.get("pipeline_components") is None:
        return kwargs
    if node_type not in QWEN_T2I_BUNDLE_INPUTS:
        raise ValueError("This native action does not declare a component bundle projection.")
    bundle = kwargs["pipeline_components"]
    token = require_component_binding(
        bundle, label="pipeline component bundle", expected_model_type=QWEN_T2I_BUNDLE_PIPELINE,
        expected_role="pipeline_components",
    )
    if node_type == "denoise" and any(kwargs.get(name) is not None for name in (
        "image_latents", "image_latents_with_strength", "controlnet_bundle", "ip_adapter", ROUTE_STATE_INPUT,
        "mask", "masked_image_latents", "control_image_latents", "control_mode", "image_embeds", "image_condition_latents",
    )):
        raise ValueError("The native component bundle facade currently declares only Qwen text-to-image.")
    identity = {name: bundle.get(name) for name in _IDENTITY_FIELDS}
    result = dict(kwargs)
    result.pop("pipeline_components")
    for field, names in QWEN_T2I_BUNDLE_INPUTS[node_type].items():
        for name in names:
            if not isinstance(bundle.get(name), dict):
                raise ValueError(f"Pipeline component bundle is missing required member '{name}'.")
            resolve_managed_component_by_id(component_manager, bundle[name], label=f"Bundle {name}")
        if field == "text_encoders":
            projection = {name: dict(bundle[name]) for name in names}
        else:
            projection = dict(bundle[names[0]])
        projection.update(identity)
        output, role = _OUTPUTS[field]
        bind_loader_outputs({output: projection}, token)
        override = kwargs.get(field)
        if override is not None:
            require_component_binding(override, label=field, expected_model_type=QWEN_T2I_BUNDLE_PIPELINE,
                                      expected_token=token, expected_role=role)
            expected_ids = collect_model_ids({field: projection}, target_key_names=(field,), target_model_names=names)
            override_ids = collect_model_ids({field: override}, target_key_names=(field,), target_model_names=names)
            if sorted(override_ids) != sorted(expected_ids):
                raise ValueError(f"Connected {field} does not match the component bundle's exact managed members.")
            result[field] = override
        else:
            result[field] = projection
    return result
