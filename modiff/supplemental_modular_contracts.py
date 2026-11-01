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
PIKA_BLOCK_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "data/pika-h3-flex-block-contract.v1.json"
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


def _contract_hash(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _reviewed_pika_source_document():
    raw = PIKA_CONTRACT_PATH.read_bytes()
    if len(raw) > 512 * 1024:
        raise ModularWorkflowContractError('The supplemental Modular contract exceeds its bound.')
    document = json.loads(raw)
    validate_supplemental_modular_contract(document)
    return document


def validate_supplemental_modular_blocks(document):
    """Validate the separate source companion without relabelling old blocks."""
    keys = {'schemaVersion', 'kind', 'source', 'runtimeProfileId', 'workflowContractHash',
            'blocks', 'scope', 'contentHash'}
    if (type(document) is not dict or set(document) != keys
            or type(document['schemaVersion']) is not int or document['schemaVersion'] != 1
            or document['kind'] != 'official-source-block-supplement'
            or document['runtimeProfileId'] != PIKA_RUNTIME_PROFILE
            or document['scope'] != 'Source-generated structural companion only; no execution or hardware qualification.'):
        raise ModularWorkflowContractError('The supplemental block contract envelope is invalid.')
    try:
        encoded = json.dumps(document, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ModularWorkflowContractError('The supplemental block contract must be finite JSON.') from error
    if len(encoded.encode('utf-8')) > 2 * 1024**2:
        raise ModularWorkflowContractError('The supplemental block contract exceeds its bound.')
    document = json.loads(encoded)
    source_document = _reviewed_pika_source_document()
    if (document['source'] != source_document['source']
            or document['workflowContractHash'] != _contract_hash(source_document['contract'])):
        raise ModularWorkflowContractError('The supplemental block contract has substituted source provenance.')
    blocks = document['blocks']
    if type(blocks) is not dict or type(blocks.get('schemaVersion')) is not int:
        raise ModularWorkflowContractError('The supplemental block snapshot is invalid.')
    from modiff.modular_block_contracts import validate_modular_block_snapshot
    validate_modular_block_snapshot(blocks, workflow_snapshot={'contracts': [source_document['contract']]},
        source_revision=PIKA_SOURCE_REVISION)
    if document['contentHash'] != _contract_hash({key: value for key, value in document.items() if key != 'contentHash'}):
        raise ModularWorkflowContractError('The supplemental block contract content hash is invalid.')
    return deepcopy(document)


def _pika_block_document(blocks):
    from modiff.modular_block_contracts import MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION
    source = _reviewed_pika_source_document()
    body = {'schemaVersion': 1, 'kind': 'official-source-block-supplement',
        'source': source['source'], 'runtimeProfileId': PIKA_RUNTIME_PROFILE,
        'workflowContractHash': _contract_hash(source['contract']),
        'blocks': {'schemaVersion': MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION,
            'diffusersRevision': PIKA_SOURCE_REVISION, **blocks},
        'scope': 'Source-generated structural companion only; no execution or hardware qualification.'}
    return validate_supplemental_modular_blocks({**body, 'contentHash': _contract_hash(body)})


@lru_cache(maxsize=1)
def _pika_blocks():
    raw = PIKA_BLOCK_CONTRACT_PATH.read_bytes()
    if len(raw) > 2 * 1024**2:
        raise ModularWorkflowContractError('The supplemental block contract exceeds its bound.')
    return validate_supplemental_modular_blocks(json.loads(raw))


def reviewed_supplemental_modular_blocks(pipeline_class):
    if pipeline_class != PIKA_PIPELINE:
        return None
    return deepcopy(_pika_blocks())


def reproduce_installed_pika_blocks():
    """Explicit no-weight reproduction through the exact already sealed provider."""
    contract = reproduce_installed_pika_contract()
    if contract != _reviewed_pika_source_document()['contract']:
        raise RuntimeError('The installed Pika workflow differs from its reviewed source contract.')
    from diffusers import PikaH3FlexBlocks
    from modiff.modular_block_contracts import build_modular_block_contracts
    return _pika_block_document(build_modular_block_contracts(PikaH3FlexBlocks().init_pipeline()))


def reviewed_modular_source_supplements():
    """Explicit approved source groups for pure-data authoring discovery."""
    document = reviewed_supplemental_modular_blocks(PIKA_PIPELINE)
    return [{'libraryRevision': document['source']['revision'],
        'runtimeProfileId': document['runtimeProfileId'],
        'workflowContractHash': document['workflowContractHash'],
        'blockContractHash': document['contentHash'],
        'contracts': [reviewed_supplemental_modular_contract(PIKA_PIPELINE)],
        'blocks': document['blocks']}]


def reviewed_modular_source_providers():
    """Lightweight declared providers, independent of installed model classes."""
    return {PIKA_PIPELINE: PIKA_RUNTIME_PROFILE}
