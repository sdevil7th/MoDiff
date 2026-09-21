"""Authoring declarations for existing actions that own and execute a model.

An integrated action remains one ordinary node. This module constructs neither
models nor an alternate execution graph, and never resolves artifact files.
"""

from copy import deepcopy

from modiff.operation_contracts import build_pipeline_operation_contract
from modiff.upscaler_contracts import REAL_ESRGAN_X2_REPO, real_esrgan_x2_model_selection


_IMAGE_UPSCALE = ("SpandrelImageUpscaleV1", "image_upscale", "modules.Spandrel.Upscaler")


def get_integrated_operation_contracts(modules):
    pipeline, task, key = _IMAGE_UPSCALE
    contract = build_pipeline_operation_contract(
        modules, pipeline_class=pipeline, task=task,
        operation_id="image.upscale", node_key=key,
    )
    if contract is None:
        return []
    contract.update(nodeType="integrated", decomposition="integrated")
    return [contract]


def integrated_operation_values(contract, *, profile=None):
    identity = (contract["binding"]["pipelineClass"], contract["task"], contract["nodeKey"])
    if identity != _IMAGE_UPSCALE:
        raise ValueError("No reviewed integrated operation binding.")
    if profile is not None and (
        profile.execution_path != "spandrel-image-upscale"
        or profile.default_repo != REAL_ESRGAN_X2_REPO
    ):
        raise ValueError("The model profile does not match the integrated operation artifact.")
    return deepcopy({"model_id": real_esrgan_x2_model_selection()})
