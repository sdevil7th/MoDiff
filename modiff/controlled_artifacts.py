"""Exact executable receipts for Studio-controlled auxiliary model artifacts.

Client receipt claims are never authoritative.  The server derives these
receipts from executable graph paths immediately before admission, resolves
reviewed Hub revisions, and hashes single-file upscalers from the managed
cache (or their explicit local file) before execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from modiff.auxiliary_lora import controlled_lora_receipts_from_graph
from modiff.model_artifact_catalog import resolve_model_revision


MAX_CONTROLLED_ARTIFACTS = 32
MAX_ARTIFACT_RECEIPT_BYTES = 32 * 1024
_EXACT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_EXACT_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PIPELINE_CLASS = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,255}$")
_MODELISH_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pt", ".pth", ".safetensors"}
_CONTROLLED_PIPELINE_CONTRACTS = {
    ("modules.DiffusersAudio", "LoadPipeline"),
    ("modules.DiffusersVideo", "LoadPipeline"),
}
_CONTROLLED_UPSCALER_CONTRACT = ("modules.Spandrel", "Upscaler")
_CONTROLLED_IMAGE_PIPELINE_CONTRACT = ("modules.DiffusersImage", "LoadPipeline")


@dataclass(frozen=True)
class ResolvedUpscalerArtifact:
    path: Path
    receipt: dict[str, Any]


def _graph_param_value(node: Mapping[str, Any], key: str, default: Any = None) -> Any:
    params = node.get("params")
    if not isinstance(params, Mapping):
        return default
    param = params.get(key)
    if not isinstance(param, Mapping):
        return default
    return param.get("value", param.get("default", default))


def _executable_node_ids(graph: Mapping[str, Any]) -> list[str]:
    paths = graph.get("paths")
    if not isinstance(paths, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not isinstance(path, list):
            continue
        for raw_node_id in path:
            node_id = str(raw_node_id)
            if node_id not in seen:
                seen.add(node_id)
                output.append(node_id)
    return output


def _exact_repository(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or value.count("/") != 1:
        raise ValueError("A controlled Hub artifact requires an exact namespace/repository ID.")
    from utils.huggingface import validate_hf_repo_id

    try:
        validate_hf_repo_id(value)
    except Exception as error:
        raise ValueError("A controlled Hub artifact requires a valid namespace/repository ID.") from error
    return value


def _exact_revision(value: Any) -> str:
    if not isinstance(value, str) or value != value.strip() or not _EXACT_COMMIT.fullmatch(value):
        raise ValueError("A controlled Hub artifact requires a lowercase 40-character commit revision.")
    return value


def _exact_sha256(value: Any, *, required: bool) -> str | None:
    if value in (None, "") and not required:
        return None
    if not isinstance(value, str) or value != value.strip() or not _EXACT_SHA256.fullmatch(value):
        raise ValueError("A controlled artifact SHA-256 must contain 64 lowercase hexadecimal digits.")
    return value


def _exact_weight_name(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("A controlled single-file artifact requires an exact relative filename.")
    if len(value) > 1024 or "\\" in value or ":" in value or "\x00" in value or value.startswith("/"):
        raise ValueError("A controlled single-file artifact must use a safe relative filename.")
    parts = value.split("/")
    if len(parts) > 64 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("A controlled single-file artifact must stay inside its selected repository.")
    if PurePosixPath(value).suffix.lower() not in _MODELISH_SUFFIXES:
        raise ValueError("A controlled single-file artifact must use a reviewed model-file extension.")
    return value


def _sha256_file(path: Path) -> str:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as error:
        raise FileNotFoundError("Controlled artifact bytes could not be read.") from error
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field, None) != getattr(after, field, None) for field in identity_fields):
        raise ValueError("Controlled artifact bytes changed while their SHA-256 was being verified.")
    return digest.hexdigest()


def _canonical_digest(value: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (RecursionError, TypeError, ValueError) as error:
        raise ValueError("Controlled artifact receipt must contain finite JSON values only.") from error
    if len(encoded.encode("utf-8")) > MAX_ARTIFACT_RECEIPT_BYTES:
        raise ValueError("Controlled artifact receipt exceeds its bounded size.")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _managed_hub_file(repository: str, revision: str, weight_name: str) -> Path:
    from utils import huggingface as huggingface_utils

    cached = huggingface_utils.cached_file_path(repository, weight_name, revision=revision)
    if not cached:
        raise FileNotFoundError("The exact controlled Hub artifact is not installed.")
    cached_path = Path(cached).expanduser()
    if not cached_path.is_absolute():
        raise ValueError("Installed controlled Hub cache entries must use absolute paths.")

    configured_root = huggingface_utils.CONFIG.hf["cache_dir"] or huggingface_utils.HUGGINGFACE_HUB_CACHE
    lexical_root = Path(os.path.abspath(Path(configured_root).expanduser()))
    lexical_alias = Path(os.path.abspath(cached_path))
    expected_alias = (
        lexical_root
        / f"models--{repository.replace('/', '--')}"
        / "snapshots"
        / revision
        / Path(*PurePosixPath(weight_name).parts)
    )
    if os.path.normcase(str(lexical_alias)) != os.path.normcase(str(expected_alias)):
        raise ValueError("Controlled Hub cache lookup did not preserve the exact repository snapshot alias.")
    try:
        cache_root = lexical_root.resolve(strict=True)
        repo_root = (cache_root / f"models--{repository.replace('/', '--')}").resolve(strict=True)
        snapshot_root = (repo_root / "snapshots" / revision).resolve(strict=True)
        if repo_root != cache_root / f"models--{repository.replace('/', '--')}":
            raise ValueError("Managed Hub repository roots cannot redirect to another location.")
        if snapshot_root != repo_root / "snapshots" / revision:
            raise ValueError("Managed Hub snapshots cannot redirect to another revision.")
        alias = snapshot_root.joinpath(*PurePosixPath(weight_name).parts)
        if alias.parent.resolve(strict=True) != alias.parent or not alias.is_file():
            raise ValueError("Controlled Hub snapshot subdirectories cannot redirect the artifact alias.")
        resolved = huggingface_utils.resolve_managed_hf_cache_file(alias)
        resolved.relative_to(repo_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Installed controlled Hub artifact is outside its exact managed snapshot.") from error
    return resolved


def _selection(value: Any) -> tuple[str, str, Mapping[str, Any]]:
    if isinstance(value, str):
        if not value or value != value.strip():
            raise ValueError("A controlled model selection must be an exact nonblank string.")
        return "local", value, {}
    if not isinstance(value, Mapping):
        raise TypeError("A controlled model selection must provide source and value fields.")
    source = value.get("source")
    selected = value.get("value")
    if source not in {"hub", "local"} or not isinstance(selected, str) or not selected or selected != selected.strip():
        raise ValueError("A controlled model selection requires an exact hub or local value.")
    return source, selected, value


def resolve_upscaler_artifact(selection: Any) -> ResolvedUpscalerArtifact:
    """Resolve and rehash one generic Spandrel model selection."""

    source, selected, metadata = _selection(selection)
    expected_sha256 = _exact_sha256(metadata.get("sha256"), required=False)
    expected_size = metadata.get("byteSize")
    if expected_size is not None and (
        isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0
    ):
        raise ValueError("Controlled artifact byteSize must be a positive integer.")

    if source == "hub":
        parts = selected.split("/")
        if len(parts) < 3:
            raise ValueError("A controlled Hub upscaler must include repository and filename.")
        repository = _exact_repository("/".join(parts[:2]))
        weight_name = _exact_weight_name("/".join(parts[2:]))
        revision = metadata.get("revision") or resolve_model_revision(repository, source="hub")
        revision = _exact_revision(revision)
        path = _managed_hub_file(repository, revision, weight_name)
        safe_artifact = {
            "source": "hub",
            "repository": repository,
            "revision": revision,
            "weightName": weight_name,
        }
    else:
        if metadata.get("revision") not in (None, ""):
            raise ValueError("A local controlled artifact cannot carry a Hub revision.")
        raw_path = Path(selected).expanduser()
        if not raw_path.is_absolute():
            from modiff.config import CONFIG

            raw_path = Path(CONFIG.paths["models"]) / raw_path
        try:
            path = raw_path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise FileNotFoundError("The selected local controlled artifact does not exist.") from error
        if not path.is_file() or path.suffix.lower() not in _MODELISH_SUFFIXES:
            raise ValueError("The selected local controlled artifact must be a model file.")
        weight_name = path.name
        safe_artifact = {"source": "local", "weightName": weight_name}

    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise ValueError("Controlled artifact bytes do not match the declared size.")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("Controlled artifact bytes do not match the declared SHA-256.")
    safe_artifact["sha256"] = actual_sha256
    payload = {
        "schemaVersion": 1,
        "kind": "spandrel_upscaler",
        "module": _CONTROLLED_UPSCALER_CONTRACT[0],
        "action": _CONTROLLED_UPSCALER_CONTRACT[1],
        "artifact": safe_artifact,
    }
    return ResolvedUpscalerArtifact(
        path=path,
        receipt={**payload, "descriptorSha256": _canonical_digest(payload)},
    )


def _pipeline_receipt(node: Mapping[str, Any]) -> dict[str, Any]:
    source, selected, metadata = _selection(_graph_param_value(node, "model_id"))
    if source != "hub":
        raise ValueError("Controlled auxiliary Diffusers pipelines require an immutable Hub artifact.")
    repository = _exact_repository(selected)
    pipeline_class = _graph_param_value(node, "pipeline_class")
    if not isinstance(pipeline_class, str) or not _PIPELINE_CLASS.fullmatch(pipeline_class):
        raise ValueError("Controlled auxiliary Diffusers pipelines require an exact reviewed class.")
    revision = _graph_param_value(node, "revision", metadata.get("revision"))
    revision = _exact_revision(resolve_model_revision(repository, revision, source="hub"))
    payload = {
        "schemaVersion": 1,
        "kind": "diffusers_pipeline",
        "module": node.get("module"),
        "action": node.get("action"),
        "artifact": {"source": "hub", "repository": repository, "revision": revision},
        "pipelineClass": pipeline_class,
    }
    return {**payload, "descriptorSha256": _canonical_digest(payload)}


def _image_conditioning_receipt(node: Mapping[str, Any]) -> dict[str, Any] | None:
    """Bind an assembled image pipeline's auxiliary component to admission."""

    from modules.DiffusersImage.main import (
        get_image_pipeline_adapter,
        repo_value,
        resolve_image_conditioning_selection,
    )

    pipeline_class = _graph_param_value(node, "pipeline_class")
    adapter = get_image_pipeline_adapter(pipeline_class)
    raw_kind = _graph_param_value(node, "conditioning_kind", "none")
    if adapter.conditioning_kind is None and raw_kind in (None, "", "none"):
        return None
    selection, revision = resolve_image_conditioning_selection(
        adapter,
        raw_kind,
        _graph_param_value(node, "conditioning_model_id"),
        _graph_param_value(node, "conditioning_revision"),
    )
    if selection is None or revision is None:
        raise ValueError("A conditioned image pipeline requires one exact auxiliary artifact.")
    payload = {
        "schemaVersion": 1,
        "kind": "diffusers_conditioning_component",
        "module": node.get("module"),
        "action": node.get("action"),
        "artifact": {
            "source": "hub",
            "repository": repo_value(selection),
            "revision": revision,
        },
        "conditioningKind": adapter.conditioning_kind,
        "componentClass": adapter.conditioning_component_class,
        "componentParameter": adapter.conditioning_component_parameter,
        "pipelineClass": adapter.pipeline_class,
        "safeSerializationRequired": True,
    }
    return {**payload, "descriptorSha256": _canonical_digest(payload)}


def _is_primary_pipeline(node: Mapping[str, Any], primary_candidate: Mapping[str, Any] | None) -> bool:
    if not isinstance(primary_candidate, Mapping):
        return False
    return (
        node.get("module") == primary_candidate.get("loaderModule")
        and node.get("action") == primary_candidate.get("loaderAction")
        and _graph_param_value(node, "pipeline_class") == primary_candidate.get("pipelineClass")
    )


def controlled_artifact_receipts_from_graph(
    graph: Any,
    *,
    primary_candidate: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return server-derived receipts for every executable controlled artifact."""

    if not isinstance(graph, Mapping):
        return []
    nodes = graph.get("nodes")
    if not isinstance(nodes, Mapping):
        return []
    receipts = controlled_lora_receipts_from_graph(graph)
    nodes_by_id = {str(node_id): node for node_id, node in nodes.items()}
    executable_nodes = [
        nodes_by_id[node_id]
        for node_id in _executable_node_ids(graph)
        if isinstance(nodes_by_id.get(node_id), Mapping)
    ]
    pipeline_nodes = [
        node
        for node in executable_nodes
        if (node.get("module"), node.get("action")) in _CONTROLLED_PIPELINE_CONTRACTS
    ]
    for node in executable_nodes:
        contract = (node.get("module"), node.get("action"))
        if contract == _CONTROLLED_UPSCALER_CONTRACT:
            receipts.append(resolve_upscaler_artifact(_graph_param_value(node, "model_id")).receipt)
        elif contract == _CONTROLLED_IMAGE_PIPELINE_CONTRACT:
            receipt = _image_conditioning_receipt(node)
            if receipt is not None:
                receipts.append(receipt)
        elif contract in _CONTROLLED_PIPELINE_CONTRACTS and (
            not _is_primary_pipeline(node, primary_candidate)
            and (isinstance(primary_candidate, Mapping) or len(pipeline_nodes) > 1)
        ):
            receipts.append(_pipeline_receipt(node))
    if len(receipts) > MAX_CONTROLLED_ARTIFACTS:
        raise ValueError(f"At most {MAX_CONTROLLED_ARTIFACTS} executable controlled artifacts are supported.")
    return receipts
