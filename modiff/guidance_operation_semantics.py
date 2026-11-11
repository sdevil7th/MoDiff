"""Reviewed authoring meanings for existing guidance controls.

This is a metadata projection, not another guider registry or runtime. Unknown
families, missing fields and other guider techniques confer no transfer authority.
The pinned ClassifierFreeGuidance constructor/forward owns execution, including
neutral scales and start/stop windows; these declarations never change its inputs.
"""


_CFG_PIPELINES = frozenset({
    "StableDiffusionXLModularPipeline", "QwenImageModularPipeline",
    "QwenImageEditModularPipeline", "QwenImageEditPlusModularPipeline",
    "QwenImageLayeredModularPipeline", "ZImageModularPipeline",
    "FluxModularPipeline", "FluxKontextModularPipeline",
})
_CFG_FIELDS = {
    "guider": ("technique", {"string"}),
    "guidance_scale": ("scale", {"float"}),
    "guidance_rescale": ("rescale", {"float"}),
    "enabled": ("enabled", {"boolean"}),
    "use_original_formulation": ("formulation", {"boolean"}),
    "start": ("start", {"float"}),
    "stop": ("stop", {"float"}),
}
_EMBEDDED_SCOPES = {
    "FluxModularPipeline": "diffusers.flux1.embedded.v1",
    "FluxKontextModularPipeline": "diffusers.flux1.embedded.v1",
    "Flux2ModularPipeline": "diffusers.flux2.embedded.v1",
}
_QWEN_NORMALIZED_CFG = frozenset({
    "QwenImageModularPipeline", "QwenImageEditModularPipeline",
    "QwenImageEditPlusModularPipeline", "QwenImageLayeredModularPipeline",
    "QwenImagePipeline", "QwenImageImg2ImgPipeline", "QwenImageInpaintPipeline",
    "QwenImageEditPipeline", "QwenImageEditPlusPipeline", "QwenImageEditInpaintPipeline",
    "QwenImageControlNetPipeline",
})
_STANDARD_FLUX_CFG = frozenset({
    "FluxPipeline", "FluxImg2ImgPipeline", "FluxInpaintPipeline",
    "FluxKontextPipeline", "FluxKontextInpaintPipeline",
})
_STANDARD_EMBEDDED = {
    "FluxFillPipeline": "diffusers.flux1.fill_embedded.v1",
    "FluxControlPipeline": "diffusers.flux1.control_embedded.v1",
    "FluxControlImg2ImgPipeline": "diffusers.flux1.control_embedded.v1",
    "FluxControlInpaintPipeline": "diffusers.flux1.control_embedded.v1",
    "FluxReduxPipeline": "diffusers.flux1.embedded.v1",
    "FluxControlNetPipeline": "diffusers.flux1.embedded.v1",
    "FluxControlNetImg2ImgPipeline": "diffusers.flux1.embedded.v1",
    "FluxControlNetInpaintPipeline": "diffusers.flux1.embedded.v1",
    "Flux2Pipeline": "diffusers.flux2.embedded.v1",
}
_STANDARD_NODES = frozenset(f"modules.DiffusersImage.{action}" for action in (
    "Generate", "Edit", "Inpaint", "ControlGenerate", "ControlEdit", "ControlInpaint", "LayerDecompose",
))
# Standard SD/SDXL can use the same scale either for CFG or for a UNet
# time-conditioning embedding. Their pinned do_classifier_free_guidance also
# checks unet.config.time_cond_proj_dim; a class/threshold declaration cannot
# establish that selected-model predicate, so those controls stay unresolved.


def _cfg_control(pipeline, parameter, *, scale_field="guidance_scale", native=False, selector=None):
    scope = "diffusers.qwen.normalized_cfg.v1" if pipeline in _QWEN_NORMALIZED_CFG else "diffusers.classifier_free.v1"
    control = {
        "technique": "classifier_free", "parameter": parameter,
        "compatibilityScope": scope, "scaleMeaning": "cfg_prediction_mix",
        "enabled": "boolean_field" if native else "scale_gt_one",
        "enabledField": "enabled" if native else scale_field,
        "formulation": "boolean_field" if native else "diffusers",
        "formulationField": "use_original_formulation" if native else None,
        "selectorField": "guider" if native else (selector[0] if selector else None),
        "selectorValue": "ClassifierFreeGuidance" if native else (selector[1] if selector else None),
        "negativeConditioning": "pipeline_scoped_when_enabled", "negativeConditioningScope": pipeline,
    }
    if not native:
        # The image adapter normalizes None to "" and forwards empty strings.
        # The pinned pipelines test `is not None`, not truthiness. A connected
        # or opaque condition is still unresolved during authoring inspection.
        control.update(negativePromptField="negative_prompt", negativePromptPolicy="empty_string_is_condition")
    return control


def _embedded_control(scope, *, selector=None):
    return {
        "technique": "embedded_distilled", "parameter": "scale",
        "compatibilityScope": scope, "scaleMeaning": "distilled_model_embedding",
        "enabled": "model_config", "enabledField": None,
        "formulation": "embedded", "formulationField": None,
        "selectorField": selector[0] if selector else None, "selectorValue": selector[1] if selector else None,
        "negativeConditioning": "none", "negativeConditioningScope": None,
    }


def _standard_controls(pipeline, ports):
    if "guidance_scale" not in ports or ports["guidance_scale"]["types"] != ["float"]:
        return {}
    controls = {}
    negative = ports.get("negative_prompt", {}).get("types") in (["string"], ["text"], ["str"])
    override = ports.get("use_guidance_scale_2", {}).get("types") in (["bool"], ["boolean"])
    secondary = ports.get("guidance_scale_2", {}).get("types") == ["float"] and override
    if pipeline in (_QWEN_NORMALIZED_CFG - _CFG_PIPELINES) | _STANDARD_FLUX_CFG and negative:
        controls["guidance_scale"] = _cfg_control(pipeline, "scale")
    elif (pipeline == "QwenImageLayeredPipeline" and negative
          and ports.get("cfg_normalize", {}).get("types") in (["bool"], ["boolean"])):
        # Only the adapter's explicit unnormalized variant is comparable to
        # plain CFG. Normalized Layered remains unresolved under this selector.
        controls["guidance_scale"] = _cfg_control(pipeline, "scale", selector=("cfg_normalize", False))
    if pipeline in _STANDARD_FLUX_CFG and secondary:
        controls["guidance_scale_2"] = _embedded_control(
            "diffusers.flux1.embedded.v1", selector=("use_guidance_scale_2", True),
        )
    if pipeline in _STANDARD_EMBEDDED:
        controls["guidance_scale"] = _embedded_control(_STANDARD_EMBEDDED[pipeline])
    if pipeline == "FluxControlNetPipeline" and negative and secondary:
        # This reviewed adapter reverses the primary/secondary aliases.
        controls["guidance_scale_2"] = _cfg_control(
            pipeline, "scale", scale_field="guidance_scale_2", selector=("use_guidance_scale_2", True),
        )
    return controls


def reviewed_guidance_controls(contract):
    """Return complete control declarations only for exact reviewed executors.

    Selector/enabled/formulation references name actual input fields, rather than
    inferring compatibility from a label. A ClassifierFreeGuidance declaration is
    conditional on the actual ``guider`` value; alternate techniques stay opaque.
    Negative tensor scope is pipeline-specific even when scalar CFG controls share
    mathematical meaning. Embedded guidance additionally needs model config: a
    class-level FLUX declaration cannot establish that Schnell consumes the scale.
    """
    pipeline = contract["pipelineClass"]
    node_key, operation = contract.get("nodeKey"), contract.get("operationId")
    cfg = node_key == "modules.ModularDiffusers.Guider" and operation == "diffusion.guidance" and pipeline in _CFG_PIPELINES
    embedded = (node_key == "modules.ModularDiffusers.Denoise"
                and operation == "diffusion.denoise" and pipeline in _EMBEDDED_SCOPES)
    standard = node_key in _STANDARD_NODES and operation in {"diffusion.generate_image", "diffusion.decompose_layers"}
    if not cfg and not embedded and not standard:
        return {}
    ports = {port["name"]: port for port in contract["ports"]
             if port.get("direction") == "input" and isinstance(port.get("name"), str)}
    if cfg:
        if any(name not in ports or set(ports[name]["types"]) != types
               for name, (_parameter, types) in _CFG_FIELDS.items()):
            return {}
        return {name: _cfg_control(pipeline, parameter, native=True)
                for name, (parameter, _types) in _CFG_FIELDS.items()}
    if (embedded
            and "guidance_scale" in ports and ports["guidance_scale"]["types"] == ["float"]):
        return {"guidance_scale": _embedded_control(_EMBEDDED_SCOPES[pipeline])}
    if standard:
        return _standard_controls(pipeline, ports)
    return {}
