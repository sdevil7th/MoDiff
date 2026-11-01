"""Reviewed newer-source declarations, without rewriting historical catalog provenance."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from modiff.modular_workflow_discovery import ModularWorkflowContractError, validate_modular_workflow_contract


PIKA_PIPELINE = "PikaH3FlexModularPipeline"
PIKA_SOURCE_REVISION = "d0b9b1b5a540e5f9df2f8ec236347bf9c5aa7d1f"
PIKA_SOURCE_TREE = "bf0c806c8af9af94867b1e8069e6230b71feb3d8"
PIKA_RUNTIME_PROFILE = "huggingface-diffusers-video-d0b9b1b"
PIKA_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "data/pika-h3-flex-workflow-contract.v1.json"
PIKA_WORKFLOW_TASKS = {
    "t2va": "text_to_video_with_audio",
    "fl2va": "first_last_frame_to_video_with_audio",
    "ref2va": "reference_to_video_with_audio",
}
_SOURCE_FILES = {
    *("src/diffusers/modular_pipelines/pika_h3_flex/" + name + ".py" for name in (
        "__init__", "modular_pipeline", "modular_blocks_pika_h3_flex", "before_encoder", "before_denoise",
        "encoders", "denoise", "decoders", "references",
    )),
    "src/diffusers/schedulers/scheduling_minimax_h3.py", "src/diffusers/loaders/lora_pipeline.py",
}


def validate_supplemental_modular_contract(document):
    """Admit only the declared data shape; this grants no execution authority."""
    if (type(document) is not dict
            or set(document) != {"schemaVersion", "kind", "source", "runtimeProfileId", "contract", "scope"}
            or type(document["schemaVersion"]) is not int or document["schemaVersion"] != 1
            or document["kind"] != "official-source-supplement"
            or document["runtimeProfileId"] != PIKA_RUNTIME_PROFILE
            or type(document["scope"]) is not str or len(document["scope"]) > 512):
        raise ModularWorkflowContractError("The supplemental Modular contract envelope is invalid.")
    source = document["source"]
    if (type(source) is not dict or set(source) != {"repository", "revision", "sourceTree", "license", "files"}
            or source["repository"] != "https://github.com/huggingface/diffusers.git"
            or source["revision"] != PIKA_SOURCE_REVISION or source["sourceTree"] != PIKA_SOURCE_TREE
            or source["license"] != "Apache-2.0" or type(source["files"]) is not dict
            or set(source["files"]) != _SOURCE_FILES
            or any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in source["files"].values())):
        raise ModularWorkflowContractError("The supplemental Modular contract has unreviewed source provenance.")
    contract = validate_modular_workflow_contract(document["contract"])
    if (contract["pipelineClass"] != PIKA_PIPELINE or contract["blocksClass"] != "PikaH3FlexBlocks"
            or {row["id"]: row["taskId"] for row in contract["workflows"]} != PIKA_WORKFLOW_TASKS
            or any({"num_inference_steps", "guidance_scale", "negative_prompt"}
                   & {field["name"] for field in row["inputs"]} for row in contract["workflows"])):
        raise ModularWorkflowContractError("The supplemental Pika workflow identity or fixed sampler is invalid.")
    return contract


@lru_cache(maxsize=1)
def _pika_contract():
    raw = PIKA_CONTRACT_PATH.read_bytes()
    if len(raw) > 512 * 1024:
        raise ModularWorkflowContractError("The supplemental Modular contract exceeds its bound.")
    return validate_supplemental_modular_contract(json.loads(raw))


def reviewed_supplemental_modular_contract(pipeline_class):
    if pipeline_class != PIKA_PIPELINE:
        return None
    return deepcopy(_pika_contract())


def reproduce_installed_pika_contract():
    """Explicit no-weight audit through an already verified installed provider."""
    import sys
    from modiff.runtime_overlays import active_diffusers_provider
    provider = active_diffusers_provider()
    if (not sys.dont_write_bytecode or not provider or provider.get("profileId") != PIKA_RUNTIME_PROFILE
            or provider.get("version") != "0.42.0.dev0"):
        raise RuntimeError("Pika contract reproduction requires -B and the active exact reviewed video provider.")
    document = json.loads(PIKA_CONTRACT_PATH.read_bytes())
    validate_supplemental_modular_contract(document)
    site = Path(provider["metadataOrigin"]).parent
    for source_name, expected in document["source"]["files"].items():
        path = site / source_name.removeprefix("src/")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("The installed video provider does not match the supplemental source hashes.")
    from diffusers import PikaH3FlexBlocks
    from modiff.modular_workflow_discovery import build_modular_workflow_contract
    return build_modular_workflow_contract(PikaH3FlexBlocks().init_pipeline(), aliases=PIKA_WORKFLOW_TASKS)
