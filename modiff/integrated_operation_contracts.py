"""Authoring declarations for existing actions that own and execute a model.

An integrated action remains one ordinary node. This module constructs neither
models nor an alternate execution graph, and never resolves artifact files.
"""

from copy import deepcopy

from modiff.operation_contracts import build_pipeline_operation_contract
from modiff.upscaler_contracts import REAL_ESRGAN_X2_REPO, real_esrgan_x2_model_selection


_IMAGE_UPSCALE = ("SpandrelImageUpscaleV1", "image_upscale", "modules.Spandrel.Upscaler")
_BUILTIN_CLASS = "BuiltinImageOperationV1"
_BUILTIN_KEY = "modules.ImageOperations.ProcessImage"
_BUILTIN_FIELDS = {
    "image_adjustment": {"brightness", "contrast", "saturation", "sharpness", "gamma", "temperature", "tint"},
    "image_filter": {"filter_operation", "amount", "threshold", "seed"},
    "image_crop": {"x", "y", "width", "height"},
    "image_upscale": {"resize_width", "resize_height", "resize_fit_mode", "resize_resampling"},
    "image_stitch": {"stitch_columns", "stitch_spacing", "stitch_background", "stitch_match_size"},
    "image_tile": {"rows", "columns"},
    "image_channels": {"channel"},
    "mask_composite": {"mask", "composite_mask_channel", "composite_invert_mask"},
}


def integrated_operation_fields(contract, modules):
    if contract["pipelineClass"] != _BUILTIN_CLASS:
        return {}
    task = contract["task"]
    visible = _BUILTIN_FIELDS[task] | {"image", "output", "operation"}
    return {
        **{name: {"hidden": name not in visible} for name in modules["modules.ImageOperations"]["ProcessImage"]["params"]},
        "image": {"required": True},
        "mask": {"required": task == "mask_composite", "hidden": task != "mask_composite"},
        "operation": {"options": [task]},
    }


def get_integrated_operation_contracts(modules):
    contracts = []
    for task in _BUILTIN_FIELDS:
        binding = {"pipelineClass": _BUILTIN_CLASS, "task": task}
        if "ProcessImage" not in modules.get("modules.ImageOperations", {}):
            continue
        contract = build_pipeline_operation_contract(
            modules, pipeline_class=_BUILTIN_CLASS, task=task, operation_id="image.process",
            node_key=_BUILTIN_KEY, field_overrides=integrated_operation_fields(binding, modules),
        )
        contract.update(nodeType="integrated", decomposition="integrated")
        contracts.append(contract)
    pipeline, task, key = _IMAGE_UPSCALE
    contract = build_pipeline_operation_contract(
        modules, pipeline_class=pipeline, task=task,
        operation_id="image.upscale", node_key=key,
    )
    if contract is None:
        return contracts
    contract.update(nodeType="integrated", decomposition="integrated")
    return [*contracts, contract]


def integrated_operation_values(contract, *, profile=None):
    identity = (contract["binding"]["pipelineClass"], contract["task"], contract["nodeKey"])
    if identity[0] == _BUILTIN_CLASS and identity[1] in _BUILTIN_FIELDS and identity[2] == _BUILTIN_KEY:
        if profile is not None and (
            profile.execution_path != "builtin-image-operation"
            or profile.default_repo != "builtin://modiff/image-operations/v1"
        ):
            raise ValueError("The profile does not match the built-in operation.")
        return {"pipeline_class": _BUILTIN_CLASS, "operation": identity[1]}
    if identity != _IMAGE_UPSCALE:
        raise ValueError("No reviewed integrated operation binding.")
    if profile is not None and (
        profile.execution_path != "spandrel-image-upscale"
        or profile.default_repo != REAL_ESRGAN_X2_REPO
    ):
        raise ValueError("The model profile does not match the integrated operation artifact.")
    return deepcopy({"model_id": real_esrgan_x2_model_selection()})
