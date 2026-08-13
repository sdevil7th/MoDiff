"""MoDiff adapter for Hugging Face Modular Diffusers node metadata.

Derived from Hugging Face Diffusers'
``src/diffusers/modular_pipelines/mellon_node_utils.py`` at commit
``bb56997d4b7e87f0743f26a612f49ec4e7ce7213`` (Apache-2.0):
https://github.com/huggingface/diffusers/blob/bb56997d4b7e87f0743f26a612f49ec4e7ce7213/src/diffusers/modular_pipelines/mellon_node_utils.py

MoDiff changes the Mellon-facing names, metadata key, configuration filename,
and imports to integrate the helper with MoDiff. The executable Diffusers
dependency is pinned separately in ``pyproject.toml``.
"""

import copy
import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping

# Simple typed wrapper for parameter overrides
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import create_repo, hf_hub_download, upload_file
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    validate_repo_id,
)

from diffusers.utils import HUGGINGFACE_CO_RESOLVE_ENDPOINT
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam


logger = logging.getLogger(__name__)


MAX_MODIFF_PIPELINE_CONFIG_BYTES = 1024 * 1024
MAX_CUSTOM_PIPELINE_REPOSITORY_CHARS = 4096
MAX_LOCAL_EXECUTABLE_MANIFEST_ENTRIES = 16384
MAX_LOCAL_EXECUTABLE_MANIFEST_FILES = 256
MAX_LOCAL_EXECUTABLE_FILE_BYTES = 2 * 1024 * 1024
MAX_LOCAL_EXECUTABLE_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_LOCAL_EXECUTABLE_MANIFEST_DEPTH = 16
MAX_CUSTOM_PIPELINE_JSON_DEPTH = 16
MAX_CUSTOM_PIPELINE_JSON_VALUES = 16_384
MAX_CUSTOM_PIPELINE_JSON_CONTAINER_ITEMS = 2_048
MAX_CUSTOM_PIPELINE_JSON_STRING_CHARS = 16_384
MAX_CUSTOM_PIPELINE_ACTIONS = 128
MAX_CUSTOM_PIPELINE_PARAMS_PER_ACTION = 256
MAX_LOADER_COMPONENT_OUTPUTS = 16
MAX_LAYER_BLOCK_OPTIONS = 64
MAX_GUIDER_OPTIONS = 16
MAX_SCHEDULER_OPTIONS = 32
MAX_DENOISE_IMAGE_LATENT_DIMENSIONS = 2
SUPPORTED_DENOISE_IMAGE_LATENT_DIMENSIONS = frozenset({"height", "width"})
PROTOTYPE_SENSITIVE_FIELD_NAMES = frozenset({"__proto__", "prototype", "constructor"})
_IMMUTABLE_HUB_REVISION = re.compile(r"^[0-9a-f]{40}$")
_LOCAL_EXECUTABLE_CONFIG_FILES = {
    "config.json",
    "model_index.json",
    "modular_config.json",
    "modular_model_index.json",
}


class DuplicateConfigKeyError(ValueError):
    """Raised when a JSON object contains an ambiguous duplicate key."""


def _reject_duplicate_config_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateConfigKeyError(f"Duplicate JSON key {key!r} is not allowed")
        value[key] = item
    return value


def _reject_nonfinite_config_constant(value):
    raise ValueError(f"Non-finite JSON number {value!r} is not allowed")


def _decode_pipeline_config_bytes(raw_bytes: bytes, *, source_label: str) -> dict[str, Any]:
    try:
        decoded = raw_bytes.decode("utf-8")
        data = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_config_keys,
            parse_constant=_reject_nonfinite_config_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise EnvironmentError(
            f"The config file at '{source_label}' is not valid unambiguous UTF-8 JSON: {error}"
        ) from error
    if not isinstance(data, dict):
        raise EnvironmentError(f"The config file at '{source_label}' must contain a JSON object at its root.")
    return data


def _validate_bounded_json_structure(data: dict[str, Any], *, source_label: str) -> None:
    """Bound nested metadata before it can become client-visible node state."""

    pending = [(data, 0)]
    value_count = 0
    while pending:
        value, depth = pending.pop()
        value_count += 1
        if value_count > MAX_CUSTOM_PIPELINE_JSON_VALUES:
            raise EnvironmentError(
                f"The config file at '{source_label}' exceeds the "
                f"{MAX_CUSTOM_PIPELINE_JSON_VALUES}-value structural limit."
            )
        if depth > MAX_CUSTOM_PIPELINE_JSON_DEPTH:
            raise EnvironmentError(
                f"The config file at '{source_label}' exceeds the "
                f"{MAX_CUSTOM_PIPELINE_JSON_DEPTH}-level structural depth limit."
            )
        if isinstance(value, dict):
            if len(value) > MAX_CUSTOM_PIPELINE_JSON_CONTAINER_ITEMS:
                raise EnvironmentError(
                    f"The config file at '{source_label}' contains an object larger than the "
                    f"{MAX_CUSTOM_PIPELINE_JSON_CONTAINER_ITEMS}-item limit."
                )
            for key, item in value.items():
                if len(key) > 256:
                    raise EnvironmentError(
                        f"The config file at '{source_label}' contains a JSON key longer than 256 characters."
                    )
                pending.append((item, depth + 1))
        elif isinstance(value, list):
            if len(value) > MAX_CUSTOM_PIPELINE_JSON_CONTAINER_ITEMS:
                raise EnvironmentError(
                    f"The config file at '{source_label}' contains a list larger than the "
                    f"{MAX_CUSTOM_PIPELINE_JSON_CONTAINER_ITEMS}-item limit."
                )
            pending.extend((item, depth + 1) for item in value)
        elif isinstance(value, str) and len(value) > MAX_CUSTOM_PIPELINE_JSON_STRING_CHARS:
            raise EnvironmentError(
                f"The config file at '{source_label}' contains a string longer than the "
                f"{MAX_CUSTOM_PIPELINE_JSON_STRING_CHARS}-character limit."
            )


def _validate_declarative_field_action(
    value: Any,
    *,
    source_label: str,
    field_path: str,
    field_definitions: Mapping[str, Any],
) -> None:
    """Reject callbacks that execute code or mutate fields outside their contract."""

    if isinstance(value, str) or not isinstance(value, (dict, list)):
        raise EnvironmentError(
            f"The config file at '{source_label}' requires '{field_path}' to be a declarative JSON object or list; "
            "string callbacks are not allowed in custom pipeline sidecars."
        )

    allowed_fields = set(field_definitions)

    def validate_visibility_map(mapping: Any) -> None:
        if not isinstance(mapping, dict):
            raise EnvironmentError(
                f"The config file at '{source_label}' requires visibility data in '{field_path}' to be an object."
            )
        for targets in mapping.values():
            target_names = targets if isinstance(targets, list) else [targets]
            if any(not isinstance(target, str) or target not in allowed_fields for target in target_names):
                raise EnvironmentError(
                    f"The config file at '{source_label}' contains an unknown field target in '{field_path}'."
                )

    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            for descriptor in item:
                if not isinstance(descriptor, (dict, list)):
                    raise EnvironmentError(
                        f"The config file at '{source_label}' contains a non-declarative callback item in "
                        f"'{field_path}'; string callbacks are not allowed."
                    )
                pending.append(descriptor)
        elif isinstance(item, dict):
            if "action" in item:
                action = item["action"]
                if not isinstance(action, str) or action not in {"show", "hide", "value", "signal"}:
                    raise EnvironmentError(
                        f"The config file at '{source_label}' contains prohibited field action "
                        f"{action!r} in '{field_path}'. Custom sidecars may only declare show, hide, value, or signal."
                    )
                if action in {"show", "hide"}:
                    validate_visibility_map(item.get("data", {}))
                else:
                    target = item.get("target")
                    if not isinstance(target, str) or target not in allowed_fields:
                        raise EnvironmentError(
                            f"The config file at '{source_label}' contains an unknown field target in '{field_path}'."
                        )
                    if action == "value":
                        prop = item.get("prop", "value")
                        if prop not in {"value", "hidden", "disabled", "options", "fieldOptions", "display"}:
                            raise EnvironmentError(
                                f"The config file at '{source_label}' contains unsupported value property "
                                f"{prop!r} in '{field_path}'."
                            )
                    else:
                        target_definition = field_definitions.get(target)
                        target_display = (
                            target_definition.get("display") if isinstance(target_definition, Mapping) else None
                        )
                        if target_display in {"input", "output"}:
                            continue
                        raise EnvironmentError(
                            f"The config file at '{source_label}' requires signal target {target!r} in "
                            f"'{field_path}' to be an input or output field."
                        )
            else:
                # An action-less object is the client's visibility map.
                validate_visibility_map(item)


def _validate_pipeline_config_document(data: dict[str, Any], *, source_label: str) -> None:
    """Validate structural fields consumed before the richer P1 schema gate."""

    _validate_bounded_json_structure(data, source_label=source_label)
    for field_name in ("label", "default_repo", "default_dtype"):
        field_value = data.get(field_name, "")
        if not isinstance(field_value, str):
            raise EnvironmentError(
                f"The config file at '{source_label}' requires string field '{field_name}'."
            )
    loader_component_outputs = data.get("loader_component_outputs", [])
    if not isinstance(loader_component_outputs, list) or len(loader_component_outputs) > MAX_LOADER_COMPONENT_OUTPUTS:
        raise EnvironmentError(
            f"The config file at '{source_label}' requires at most {MAX_LOADER_COMPONENT_OUTPUTS} "
            "loader component output names."
        )
    invalid_loader_component_output = any(
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 128
        or name in PROTOTYPE_SENSITIVE_FIELD_NAMES
        for name in loader_component_outputs
    )
    if invalid_loader_component_output or len(loader_component_outputs) != len(set(loader_component_outputs)):
        raise EnvironmentError(
            f"The config file at '{source_label}' contains invalid or duplicate loader component output names."
        )
    layer_block_options = data.get("layer_block_options", [])
    if not isinstance(layer_block_options, list) or len(layer_block_options) > MAX_LAYER_BLOCK_OPTIONS:
        raise EnvironmentError(
            f"The config file at '{source_label}' requires at most {MAX_LAYER_BLOCK_OPTIONS} layer block names."
        )
    invalid_layer_block_option = any(
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 256
        or name in PROTOTYPE_SENSITIVE_FIELD_NAMES
        for name in layer_block_options
    )
    if invalid_layer_block_option or len(layer_block_options) != len(set(layer_block_options)):
        raise EnvironmentError(
            f"The config file at '{source_label}' contains invalid or duplicate layer block names."
        )
    guider_options = data.get("guider_options", [])
    if not isinstance(guider_options, list) or len(guider_options) > MAX_GUIDER_OPTIONS:
        raise EnvironmentError(
            f"The config file at '{source_label}' requires at most {MAX_GUIDER_OPTIONS} guider class names."
        )
    invalid_guider_option = any(
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 128
        or name in PROTOTYPE_SENSITIVE_FIELD_NAMES
        for name in guider_options
    )
    if invalid_guider_option or len(guider_options) != len(set(guider_options)):
        raise EnvironmentError(
            f"The config file at '{source_label}' contains invalid or duplicate guider class names."
        )
    scheduler_options = data.get("scheduler_options", [])
    if not isinstance(scheduler_options, list) or len(scheduler_options) > MAX_SCHEDULER_OPTIONS:
        raise EnvironmentError(
            f"The config file at '{source_label}' requires at most {MAX_SCHEDULER_OPTIONS} scheduler class names."
        )
    invalid_scheduler_option = any(
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 128
        or name in PROTOTYPE_SENSITIVE_FIELD_NAMES
        for name in scheduler_options
    )
    if invalid_scheduler_option or len(scheduler_options) != len(set(scheduler_options)):
        raise EnvironmentError(
            f"The config file at '{source_label}' contains invalid or duplicate scheduler class names."
        )
    denoise_image_latent_dimensions = data.get("denoise_image_latent_dimensions", [])
    if (
        not isinstance(denoise_image_latent_dimensions, list)
        or len(denoise_image_latent_dimensions) > MAX_DENOISE_IMAGE_LATENT_DIMENSIONS
    ):
        raise EnvironmentError(
            f"The config file at '{source_label}' requires at most "
            f"{MAX_DENOISE_IMAGE_LATENT_DIMENSIONS} denoise image-latent dimension names."
        )
    if (
        any(
            not isinstance(name, str) or name not in SUPPORTED_DENOISE_IMAGE_LATENT_DIMENSIONS
            for name in denoise_image_latent_dimensions
        )
        or len(denoise_image_latent_dimensions) != len(set(denoise_image_latent_dimensions))
    ):
        raise EnvironmentError(
            f"The config file at '{source_label}' contains invalid or duplicate denoise image-latent dimension names."
        )
    if "node_params" not in data or not isinstance(data["node_params"], dict) or not data["node_params"]:
        raise EnvironmentError(
            f"The config file at '{source_label}' requires a non-empty 'node_params' JSON object."
        )
    if len(data["node_params"]) > MAX_CUSTOM_PIPELINE_ACTIONS:
        raise EnvironmentError(
            f"The config file at '{source_label}' exceeds the {MAX_CUSTOM_PIPELINE_ACTIONS}-action limit."
        )

    for action_name, action in data["node_params"].items():
        if (
            not isinstance(action_name, str)
            or not action_name.strip()
            or len(action_name) > 128
            or action_name in PROTOTYPE_SENSITIVE_FIELD_NAMES
        ):
            raise EnvironmentError(
                f"The config file at '{source_label}' contains an invalid node action name."
            )
        if action is None:
            continue
        if not isinstance(action, dict):
            raise EnvironmentError(
                f"The config file at '{source_label}' requires action '{action_name}' to be a JSON object or null."
            )
        params = action.get("params")
        if not isinstance(params, dict):
            raise EnvironmentError(
                f"The config file at '{source_label}' requires action '{action_name}.params' to be a JSON object."
            )
        if len(params) > MAX_CUSTOM_PIPELINE_PARAMS_PER_ACTION:
            raise EnvironmentError(
                f"The config file at '{source_label}' action '{action_name}' exceeds the "
                f"{MAX_CUSTOM_PIPELINE_PARAMS_PER_ACTION}-parameter limit."
            )
        for param_name, param in params.items():
            if (
                not isinstance(param_name, str)
                or not param_name.strip()
                or len(param_name) > 128
                or param_name in PROTOTYPE_SENSITIVE_FIELD_NAMES
            ):
                raise EnvironmentError(
                    f"The config file at '{source_label}' contains an invalid parameter name in '{action_name}'."
                )
            if not isinstance(param, dict):
                raise EnvironmentError(
                    f"The config file at '{source_label}' requires parameter '{action_name}.{param_name}' "
                    "to be a JSON object."
                )
            for callback_name in ("onChange", "onSignal"):
                if callback_name in param:
                    _validate_declarative_field_action(
                        param[callback_name],
                        source_label=source_label,
                        field_path=f"{action_name}.{param_name}.{callback_name}",
                        field_definitions=params,
                    )
        for names_field in ("input_names", "model_input_names", "output_names"):
            names = action.get(names_field)
            if not isinstance(names, list) or any(
                not isinstance(name, str) or not name.strip() or len(name) > 128 for name in names
            ):
                raise EnvironmentError(
                    f"The config file at '{source_label}' requires '{action_name}.{names_field}' "
                    "to be a list of non-empty strings."
                )
        block_name = action.get("block_name")
        if block_name is not None and (
            not isinstance(block_name, str) or not block_name.strip() or len(block_name) > 256
        ):
            raise EnvironmentError(
                f"The config file at '{source_label}' requires '{action_name}.block_name' "
                "to be a non-empty string or null."
            )
        for optional_string in ("node_type", "label", "color"):
            if optional_string in action and not isinstance(action[optional_string], str):
                raise EnvironmentError(
                    f"The config file at '{source_label}' requires '{action_name}.{optional_string}' "
                    "to be a string."
                )


def _read_pipeline_config_bytes(config_path: Path) -> bytes:
    try:
        size = config_path.stat().st_size
    except OSError as error:
        raise EnvironmentError(f"Could not inspect Modular Diffusers config file '{config_path}': {error}") from error
    if size > MAX_MODIFF_PIPELINE_CONFIG_BYTES:
        raise EnvironmentError(
            f"The Modular Diffusers config file at '{config_path}' is larger than the "
            f"{MAX_MODIFF_PIPELINE_CONFIG_BYTES}-byte limit."
        )
    try:
        with config_path.open("rb") as reader:
            raw_bytes = reader.read(MAX_MODIFF_PIPELINE_CONFIG_BYTES + 1)
    except OSError as error:
        raise EnvironmentError(f"Could not read Modular Diffusers config file '{config_path}': {error}") from error
    if len(raw_bytes) > MAX_MODIFF_PIPELINE_CONFIG_BYTES:
        raise EnvironmentError(
            f"The Modular Diffusers config file at '{config_path}' is larger than the "
            f"{MAX_MODIFF_PIPELINE_CONFIG_BYTES}-byte limit."
        )
    return raw_bytes


def _is_executable_manifest_file(path: Path, relative_name: str) -> bool:
    return relative_name in _LOCAL_EXECUTABLE_CONFIG_FILES or path.suffix.lower() == ".py"


def _hub_snapshot_blob_root(repository_path: Path) -> Path:
    return repository_path.parent.parent / "blobs"


def _is_linked_directory(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _validate_hub_snapshot_config_path(config_path: Path, *, revision: str) -> None:
    repository_path = config_path.parent
    if (
        config_path.name != MoDiffPipelineConfig.config_name
        or repository_path.name != revision
        or repository_path.parent.name != "snapshots"
    ):
        raise EnvironmentError(
            f"The cached Hub result for revision {revision} is not contained in its exact snapshot directory."
        )
    repo_cache_root = repository_path.parent.parent
    for directory, label in (
        (repo_cache_root, "repository cache"),
        (repository_path.parent, "snapshots directory"),
        (repository_path, "exact snapshot"),
    ):
        if _is_linked_directory(directory):
            raise EnvironmentError(f"The cached Hub {label} must not be a symlink or junction: '{directory}'.")
    if config_path.is_symlink():
        try:
            blob_root = _hub_snapshot_blob_root(repository_path)
            if _is_linked_directory(blob_root):
                raise ValueError("the repository blobs directory is linked")
            target = config_path.resolve(strict=True)
            target.relative_to(blob_root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as error:
            raise EnvironmentError(
                f"The cached Hub {MoDiffPipelineConfig.config_name} symlink does not resolve inside this "
                "repository cache's blobs directory."
            ) from error


def _executable_manifest_sha256(repository_path: Path, *, source: str) -> str:
    """Detect bounded loader/control metadata drift, not weights or atomic code."""

    candidates = []
    pending = [(repository_path, 0)]
    scanned_entries = 0
    while pending:
        current_directory, depth = pending.pop()
        try:
            entries = sorted(os.scandir(current_directory), key=lambda entry: entry.name)
        except OSError as error:
            raise EnvironmentError(
                f"Could not inspect custom pipeline directory '{current_directory}': {error}"
            ) from error
        for entry in entries:
            scanned_entries += 1
            if scanned_entries > MAX_LOCAL_EXECUTABLE_MANIFEST_ENTRIES:
                raise EnvironmentError(
                    "Custom pipeline repository exceeds the "
                    f"{MAX_LOCAL_EXECUTABLE_MANIFEST_ENTRIES}-entry executable-manifest scan limit."
                )
            entry_path = Path(entry.path)
            relative_name = entry_path.relative_to(repository_path).as_posix()
            manifest_file = _is_executable_manifest_file(entry_path, relative_name)
            is_junction = bool(getattr(entry_path, "is_junction", lambda: False)())
            if entry.is_symlink() or is_junction:
                if entry.is_dir(follow_symlinks=True) or is_junction:
                    raise EnvironmentError(
                        f"Custom pipeline executable manifest does not allow linked directory '{entry_path}'."
                    )
                if manifest_file:
                    if source == "local":
                        raise EnvironmentError(
                            f"Local custom pipeline executable manifest does not allow linked file '{entry_path}'."
                        )
                    try:
                        blob_root = _hub_snapshot_blob_root(repository_path)
                        if _is_linked_directory(blob_root):
                            raise ValueError("the repository blobs directory is linked")
                        resolved_path = entry_path.resolve(strict=True)
                        resolved_path.relative_to(blob_root.resolve(strict=True))
                        if not resolved_path.is_file():
                            raise ValueError("linked executable is not a regular file")
                    except (OSError, RuntimeError, ValueError) as error:
                        raise EnvironmentError(
                            f"Cached Hub executable file '{entry_path}' does not resolve inside this repository "
                            "cache's blobs directory."
                        ) from error
                    candidates.append((entry_path, resolved_path))
                continue
            if entry.is_dir(follow_symlinks=False):
                if depth >= MAX_LOCAL_EXECUTABLE_MANIFEST_DEPTH:
                    raise EnvironmentError(
                        "Custom pipeline repository exceeds the "
                        f"{MAX_LOCAL_EXECUTABLE_MANIFEST_DEPTH}-level executable-manifest depth limit."
                    )
                pending.append((entry_path, depth + 1))
            elif entry.is_file(follow_symlinks=False) and manifest_file:
                candidates.append((entry_path, entry_path))

    if len(candidates) > MAX_LOCAL_EXECUTABLE_MANIFEST_FILES:
        raise EnvironmentError(
            "Custom pipeline repository exceeds the "
            f"{MAX_LOCAL_EXECUTABLE_MANIFEST_FILES}-file executable-manifest limit."
        )

    manifest_entries = []
    total_bytes = 0
    for display_path, resolved_path in candidates:
        relative_name = display_path.relative_to(repository_path).as_posix()
        if len(relative_name) > MAX_CUSTOM_PIPELINE_REPOSITORY_CHARS:
            raise EnvironmentError("Custom pipeline executable manifest path exceeds 4096 characters.")
        try:
            file_size = resolved_path.stat().st_size
        except OSError as error:
            raise EnvironmentError(
                f"Could not inspect custom pipeline executable file '{display_path}': {error}"
            ) from error
        if file_size > MAX_LOCAL_EXECUTABLE_FILE_BYTES:
            raise EnvironmentError(
                f"Custom pipeline executable file '{display_path}' exceeds the "
                f"{MAX_LOCAL_EXECUTABLE_FILE_BYTES}-byte limit."
            )
        try:
            with resolved_path.open("rb") as reader:
                raw_bytes = reader.read(MAX_LOCAL_EXECUTABLE_FILE_BYTES + 1)
        except OSError as error:
            raise EnvironmentError(
                f"Could not read custom pipeline executable file '{display_path}': {error}"
            ) from error
        if len(raw_bytes) > MAX_LOCAL_EXECUTABLE_FILE_BYTES:
            raise EnvironmentError(
                f"Custom pipeline executable file '{display_path}' exceeds the "
                f"{MAX_LOCAL_EXECUTABLE_FILE_BYTES}-byte limit."
            )
        if len(raw_bytes) != file_size:
            raise EnvironmentError(
                f"Custom pipeline loader metadata '{display_path}' changed while its manifest was being read. "
                "Retry after the repository cache is stable."
            )
        total_bytes += len(raw_bytes)
        if total_bytes > MAX_LOCAL_EXECUTABLE_MANIFEST_BYTES:
            raise EnvironmentError(
                "Custom pipeline executable manifest exceeds the "
                f"{MAX_LOCAL_EXECUTABLE_MANIFEST_BYTES}-byte total limit."
            )
        manifest_entries.append((relative_name, raw_bytes))

    digest = hashlib.sha256(b"modiff-executable-manifest-v2\0")
    present_names = {relative_name for relative_name, _raw_bytes in manifest_entries}
    for config_name in sorted(_LOCAL_EXECUTABLE_CONFIG_FILES):
        name_bytes = config_name.encode("utf-8")
        digest.update(b"fixed-config\0")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(b"present\0" if config_name in present_names else b"absent\0")
    for relative_name, raw_bytes in sorted(manifest_entries):
        name_bytes = relative_name.encode("utf-8")
        digest.update(b"file\0")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(raw_bytes).to_bytes(8, "big"))
        digest.update(raw_bytes)
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedMoDiffPipelineConfig:
    """A bounded sidecar plus loader-metadata drift checksum; model weights are not hashed."""

    config: "MoDiffPipelineConfig"
    raw_bytes: bytes
    sha256: str
    source: str
    repo_id: str
    revision: str | None
    executable_manifest_sha256: str
    config_path: str
    repository_path: str


def _name_to_label(name: str) -> str:
    """Convert snake_case name to Title Case label."""
    return name.replace("_", " ").title()


# Template definitions for standard diffuser pipeline parameters
MODIFF_PARAM_TEMPLATES = {
    # Image I/O
    "image": {"label": "Image", "type": "image", "display": "input", "required_block_params": ["image"]},
    "last_image": {
        "label": "Last Image",
        "type": "image",
        "display": "input",
        "required_block_params": ["last_image"],
    },
    "images": {"label": "Images", "type": "image", "display": "output", "required_block_params": ["images"]},
    "control_image": {
        "label": "Control Image",
        "type": "image",
        "display": "input",
        "required_block_params": ["control_image"],
    },
    "mask_image": {
        "label": "Mask Image",
        "type": "image",
        "display": "input",
        "required_block_params": ["mask_image"],
    },
    # Latents
    "latents": {"label": "Latents", "type": "latents", "display": "input", "required_block_params": ["latents"]},
    "image_latents": {
        "label": "Image Latents",
        "type": "latents",
        "display": "input",
        "required_block_params": ["image_latents"],
    },
    "mask": {
        "label": "Latent Mask",
        "type": "latent_mask",
        "display": "input",
        "required_block_params": ["mask"],
    },
    "masked_image_latents": {
        "label": "Masked Image Latents",
        "type": "masked_latents",
        "display": "input",
        "required_block_params": ["masked_image_latents"],
    },
    "first_frame_latents": {
        "label": "First Frame Latents",
        "type": "latents",
        "display": "input",
        "required_block_params": ["first_frame_latents"],
    },
    "image_condition_latents": {
        "label": "Image Condition Latents",
        "type": "video_condition_latents",
        "display": "output",
        "required_block_params": ["image_condition_latents"],
    },
    "latents_preview": {"label": "Latents Preview", "type": "latent", "display": "output"},
    # Image Latents with Strength
    "image_latents_with_strength": {
        "name": "image_latents",  # name is not same as template key
        "label": "Image Latents",
        "type": "latents",
        "display": "input",
        "onChange": {"false": ["height", "width"], "true": ["strength"]},
        "required_block_params": ["image_latents", "strength"],
    },
    # Embeddings
    "embeddings": {"label": "Text Embeddings", "type": "embeddings", "display": "output"},
    "image_embeds": {
        "label": "Image Embeddings",
        "type": "image_embeds",
        "display": "output",
        "required_block_params": ["image_embeds"],
    },
    # Text inputs
    "prompt": {
        "label": "Prompt",
        "type": "string",
        "display": "textarea",
        "default": "",
        "required_block_params": ["prompt"],
    },
    "negative_prompt": {
        "label": "Negative Prompt",
        "type": "string",
        "display": "textarea",
        "default": "",
        "required_block_params": ["negative_prompt"],
    },
    # Numeric params
    "guidance_scale": {
        "label": "Guidance Scale",
        "type": "float",
        "display": "slider",
        "default": 5.0,
        "min": 1.0,
        "max": 30.0,
        "step": 0.1,
    },
    "strength": {
        "label": "Strength",
        "type": "float",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required_block_params": ["strength"],
    },
    "height": {
        "label": "Height",
        "type": "int",
        "default": 1024,
        "min": 64,
        "step": 8,
        "required_block_params": ["height"],
    },
    "width": {
        "label": "Width",
        "type": "int",
        "default": 1024,
        "min": 64,
        "step": 8,
        "required_block_params": ["width"],
    },
    "seed": {
        "label": "Seed",
        "type": "int",
        "default": 0,
        "min": 0,
        "max": 4294967295,
        "display": "random",
        "required_block_params": ["generator"],
    },
    "padding_mask_crop": {
        "label": "Mask Crop Padding",
        "type": "int",
        "min": 0,
        "max": 8192,
        "step": 1,
        "required_block_params": ["padding_mask_crop"],
    },
    "num_inference_steps": {
        "label": "Steps",
        "type": "int",
        "default": 25,
        "min": 1,
        "max": 100,
        "display": "slider",
        "required_block_params": ["num_inference_steps"],
    },
    "num_frames": {
        "label": "Frames",
        "type": "int",
        "default": 81,
        "min": 1,
        "max": 480,
        "display": "slider",
        "required_block_params": ["num_frames"],
    },
    "layers": {
        "label": "Layers",
        "type": "int",
        "default": 4,
        "min": 1,
        "max": 10,
        "display": "slider",
        "required_block_params": ["layers"],
    },
    "output_type": {
        "label": "Output Type",
        "type": "dropdown",
        "default": "np",
        "options": ["np", "pil", "pt"],
    },
    # ControlNet
    "controlnet_conditioning_scale": {
        "label": "Controlnet Conditioning Scale",
        "type": "float",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required_block_params": ["controlnet_conditioning_scale"],
    },
    "control_guidance_start": {
        "label": "Control Guidance Start",
        "type": "float",
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required_block_params": ["control_guidance_start"],
    },
    "control_guidance_end": {
        "label": "Control Guidance End",
        "type": "float",
        "default": 1.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required_block_params": ["control_guidance_end"],
    },
    # Video
    "videos": {"label": "Videos", "type": "video", "display": "output", "required_block_params": ["videos"]},
    # Models
    "vae": {"label": "VAE", "type": "diffusers_auto_model", "display": "input", "required_block_params": ["vae"]},
    "image_encoder": {
        "label": "Image Encoder",
        "type": "diffusers_auto_model",
        "display": "input",
        "required_block_params": ["image_encoder"],
    },
    "unet": {"label": "Denoise Model", "type": "diffusers_auto_model", "display": "input"},
    "scheduler": {"label": "Scheduler", "type": "diffusers_auto_model", "display": "input"},
    "controlnet": {
        "label": "ControlNet Model",
        "type": "diffusers_auto_model",
        "display": "input",
        "required_block_params": ["controlnet"],
    },
    "text_encoders": {
        "label": "Text Encoders",
        "type": "diffusers_auto_models",
        "display": "input",
        "required_block_params": ["text_encoder"],
    },
    # Bundles/Custom
    "controlnet_bundle": {
        "label": "ControlNet",
        "type": "custom_controlnet",
        "display": "input",
        "required_block_params": "controlnet_image",
    },
    "ip_adapter": {"label": "IP Adapter", "type": "custom_ip_adapter", "display": "input"},
    "guider": {
        "label": "Guider",
        "type": "custom_guider",
        "display": "input",
        "onChange": {False: ["guidance_scale"], True: []},
        "required_block_params": ["guider"],
    },
    "doc": {"label": "Doc", "type": "string", "display": "output"},
    "route_state_in": {
        "label": "Route State",
        "type": "modular_route_state",
        "display": "input",
    },
    "route_state_out": {
        "label": "Route State",
        "type": "modular_route_state",
        "display": "output",
    },
}


class MoDiffParamMeta(type):
    """Metaclass that enables MoDiffParam.template_name(**overrides) syntax."""

    def __getattr__(cls, name: str):
        if name in MODIFF_PARAM_TEMPLATES:

            def factory(default=None, **overrides):
                template = MODIFF_PARAM_TEMPLATES[name]
                # Use template's name if specified, otherwise use the key
                params = {"name": template.get("name", name), **template, **overrides}
                if default is not None:
                    params["default"] = default
                return cls(**params)

            return factory

        raise AttributeError(f"type object 'MoDiffParam' has no attribute '{name}'")


@dataclass(frozen=True)
class MoDiffParam(metaclass=MoDiffParamMeta):
    """
        Parameter definition for MoDiff nodes.

        Usage:
    ```python
        # From template (standard diffuser params)
        MoDiffParam.seed()
        MoDiffParam.prompt(default="a cat")
        MoDiffParam.latents(display="output")

        # Generic inputs (for custom blocks)
        MoDiffParam.Input.slider("my_scale", default=1.0, min=0.0, max=2.0)
        MoDiffParam.Input.dropdown("mode", options=["fast", "slow"])

        # Generic outputs
        MoDiffParam.Output.image("result_images")

        # Fully custom
        MoDiffParam(name="custom", label="Custom", type="float", default=0.5)
    ```
    """

    name: str
    label: str
    type: str
    display: str | None = None
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: Any = None
    value: Any = None
    fieldOptions: dict[str, Any] | None = None
    onChange: Any = None
    onSignal: Any = None
    required_block_params: str | list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for MoDiff schema, excluding None values and internal fields."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None and k not in ("name", "required_block_params")}

    # =========================================================================
    # Input: Generic input parameter factories (for custom blocks)
    # =========================================================================
    class Input:
        """input UI elements for custom blocks."""

        @classmethod
        def image(cls, name: str) -> "MoDiffParam":
            """image input."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="image", display="input")

        @classmethod
        def textbox(cls, name: str, default: str = "") -> "MoDiffParam":
            """text input as textarea."""
            return MoDiffParam(
                name=name, label=_name_to_label(name), type="string", display="textarea", default=default
            )

        @classmethod
        def dropdown(cls, name: str, options: list[str] = None, default: str = None) -> "MoDiffParam":
            """dropdown selection."""
            if options and not default:
                default = options[0]
            if not default:
                default = ""
            if not options:
                options = [default]
            return MoDiffParam(name=name, label=_name_to_label(name), type="string", options=options, value=default)

        @classmethod
        def slider(
            cls, name: str, default: float = 0, min: float = None, max: float = None, step: float = None
        ) -> "MoDiffParam":
            """slider input."""
            is_float = isinstance(default, float) or (step is not None and isinstance(step, float))
            param_type = "float" if is_float else "int"
            if min is None:
                min = default
            if max is None:
                max = default
            if step is None:
                step = 0.01 if is_float else 1
            return MoDiffParam(
                name=name,
                label=_name_to_label(name),
                type=param_type,
                display="slider",
                default=default,
                min=min,
                max=max,
                step=step,
            )

        @classmethod
        def number(
            cls, name: str, default: float = 0, min: float = None, max: float = None, step: float = None
        ) -> "MoDiffParam":
            """number input (no slider)."""
            is_float = isinstance(default, float) or (step is not None and isinstance(step, float))
            param_type = "float" if is_float else "int"
            return MoDiffParam(
                name=name, label=_name_to_label(name), type=param_type, default=default, min=min, max=max, step=step
            )

        @classmethod
        def seed(cls, name: str = "seed", default: int = 0) -> "MoDiffParam":
            """seed input with randomize button."""
            return MoDiffParam(
                name=name,
                label=_name_to_label(name),
                type="int",
                display="random",
                default=default,
                min=0,
                max=4294967295,
            )

        @classmethod
        def checkbox(cls, name: str, default: bool = False) -> "MoDiffParam":
            """boolean checkbox."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="boolean", value=default)

        @classmethod
        def custom_type(cls, name: str, type: str) -> "MoDiffParam":
            """custom type input for node connections."""
            return MoDiffParam(name=name, label=_name_to_label(name), type=type, display="input")

        @classmethod
        def model(cls, name: str) -> "MoDiffParam":
            """model input for diffusers components."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="diffusers_auto_model", display="input")

    # =========================================================================
    # Output: Generic output parameter factories (for custom blocks)
    # =========================================================================
    class Output:
        """output UI elements for custom blocks."""

        @classmethod
        def image(cls, name: str) -> "MoDiffParam":
            """image output."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="image", display="output")

        @classmethod
        def video(cls, name: str) -> "MoDiffParam":
            """video output."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="video", display="output")

        @classmethod
        def text(cls, name: str) -> "MoDiffParam":
            """text output."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="string", display="output")

        @classmethod
        def custom_type(cls, name: str, type: str) -> "MoDiffParam":
            """custom type output for node connections."""
            return MoDiffParam(name=name, label=_name_to_label(name), type=type, display="output")

        @classmethod
        def model(cls, name: str) -> "MoDiffParam":
            """model output for diffusers components."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="diffusers_auto_model", display="output")


def input_param_to_modiff_param(input_param: "InputParam") -> MoDiffParam:
    """
    Convert an InputParam to a MoDiffParam using metadata.

    Args:
        input_param: An InputParam with optional metadata containing either:
            - {"modiff": "<type>"} for simple types (image, textbox, slider, etc.)
            - {"modiff": MoDiffParam(...)} for full control over UI configuration

    Returns:
        MoDiffParam instance
    """
    name = input_param.name
    metadata = input_param.metadata
    modiff_value = metadata.get("modiff") if metadata else None
    default = input_param.default

    # If it's already a MoDiffParam, return it directly
    if isinstance(modiff_value, MoDiffParam):
        return modiff_value

    modiff_type = modiff_value

    if modiff_type == "image":
        return MoDiffParam.Input.image(name)
    elif modiff_type == "textbox":
        return MoDiffParam.Input.textbox(name, default=default or "")
    elif modiff_type == "dropdown":
        return MoDiffParam.Input.dropdown(name, default=default or "")
    elif modiff_type == "slider":
        return MoDiffParam.Input.slider(name, default=default or 0)
    elif modiff_type == "number":
        return MoDiffParam.Input.number(name, default=default or 0)
    elif modiff_type == "seed":
        return MoDiffParam.Input.seed(name, default=default or 0)
    elif modiff_type == "checkbox":
        return MoDiffParam.Input.checkbox(name, default=default or False)
    elif modiff_type == "model":
        return MoDiffParam.Input.model(name)
    else:
        # None or unknown -> custom
        return MoDiffParam.Input.custom_type(name, type="custom")


def output_param_to_modiff_param(output_param: "OutputParam") -> MoDiffParam:
    """
    Convert an OutputParam to a MoDiffParam using metadata.

    Args:
        output_param: An OutputParam with optional metadata={"modiff": "<type>"} where type is one of:
            image, video, text, model. If metadata is None or unknown, maps to "custom".

    Returns:
        MoDiffParam instance
    """
    name = output_param.name
    metadata = output_param.metadata
    modiff_type = metadata.get("modiff") if metadata else None

    if modiff_type == "image":
        return MoDiffParam.Output.image(name)
    elif modiff_type == "video":
        return MoDiffParam.Output.video(name)
    elif modiff_type == "text":
        return MoDiffParam.Output.text(name)
    elif modiff_type == "model":
        return MoDiffParam.Output.model(name)
    else:
        # None or unknown -> custom
        return MoDiffParam.Output.custom_type(name, type="custom")


DEFAULT_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            MoDiffParam.embeddings(display="input"),
            MoDiffParam.width(),
            MoDiffParam.height(),
            MoDiffParam.seed(),
            MoDiffParam.num_inference_steps(),
            MoDiffParam.num_frames(),
            MoDiffParam.guidance_scale(),
            MoDiffParam.strength(),
            MoDiffParam.image_latents_with_strength(),
            MoDiffParam.image_latents(),
            MoDiffParam.first_frame_latents(),
            MoDiffParam.controlnet_bundle(display="input"),
        ],
        "model_inputs": [
            MoDiffParam.unet(),
            MoDiffParam.guider(),
            MoDiffParam.scheduler(),
        ],
        "outputs": [
            MoDiffParam.latents(display="output"),
            MoDiffParam.latents_preview(),
            MoDiffParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MoDiffParam.image(),
            MoDiffParam.seed(),
        ],
        "model_inputs": [
            MoDiffParam.vae(),
        ],
        "outputs": [
            MoDiffParam.image_latents(display="output"),
            MoDiffParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MoDiffParam.prompt(),
            MoDiffParam.negative_prompt(),
        ],
        "model_inputs": [
            MoDiffParam.text_encoders(),
        ],
        "outputs": [
            MoDiffParam.embeddings(display="output"),
            MoDiffParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MoDiffParam.latents(display="input"),
        ],
        "model_inputs": [
            MoDiffParam.vae(),
        ],
        "outputs": [
            MoDiffParam.images(),
            MoDiffParam.videos(),
            MoDiffParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}


def mark_required(label: str, marker: str = " *") -> str:
    """Add required marker to label if not already present."""
    if label.endswith(marker):
        return label
    return f"{label}{marker}"


def node_spec_to_modiff_dict(node_spec: dict[str, Any], node_type: str) -> dict[str, Any]:
    """
    Convert a node spec dict into MoDiff format.

    A node spec is how we define a MoDiff diffusers node in code. This function converts it into the `params` map
    format that MoDiff UI expects.

    The `params` map is a dict where keys are parameter names and values are UI configuration:
        ```python
        {"seed": {"label": "Seed", "type": "int", "default": 0}}
        ```

    For Modular MoDiff nodes, we need to distinguish:
        - `inputs`: Pipeline inputs (e.g., seed, prompt, image)
        - `model_inputs`: Model components (e.g., unet, vae, scheduler)
        - `outputs`: Node outputs (e.g., latents, images)

    The node spec also includes:
        - `required_inputs` / `required_model_inputs`: Which params are required (marked with *)
        - `block_name`: The modular pipeline block this node corresponds to on backend

    We provide factory methods for common parameters (e.g., `MoDiffParam.seed()`, `MoDiffParam.unet()`) so you don't
    have to manually specify all the UI configuration.

    Args:
        node_spec: Dict with `inputs`, `model_inputs`, `outputs` (lists of MoDiffParam),
                   plus `required_inputs`, `required_model_inputs`, `block_name`.
        node_type: The node type string (e.g., "denoise", "controlnet")

    Returns:
        Dict with:
            - `params`: Flat dict of all params in MoDiff UI format
            - `input_names`: List of input parameter names
            - `model_input_names`: List of model input parameter names
            - `output_names`: List of output parameter names
            - `block_name`: The backend block name
            - `node_type`: The node type

    Example:
        ```python
        node_spec = {
            "inputs": [MoDiffParam.seed(), MoDiffParam.prompt()],
            "model_inputs": [MoDiffParam.unet()],
            "outputs": [MoDiffParam.latents(display="output")],
            "required_inputs": ["prompt"],
            "required_model_inputs": ["unet"],
            "block_name": "denoise",
        }

        result = node_spec_to_modiff_dict(node_spec, "denoise")
        # Returns:
        # {
        #     "params": {
        #         "seed": {"label": "Seed", "type": "int", "default": 0},
        #         "prompt": {"label": "Prompt *", "type": "string", "default": ""},  # * marks required
        #         "unet": {"label": "Denoise Model *", "type": "diffusers_auto_model", "display": "input"},
        #         "latents": {"label": "Latents", "type": "latents", "display": "output"},
        #     },
        #     "input_names": ["seed", "prompt"],
        #     "model_input_names": ["unet"],
        #     "output_names": ["latents"],
        #     "block_name": "denoise",
        #     "node_type": "denoise",
        # }
        ```
    """
    params = {}
    input_names = []
    model_input_names = []
    output_names = []

    required_inputs = node_spec.get("required_inputs", [])
    required_model_inputs = node_spec.get("required_model_inputs", [])

    # Process inputs
    for p in node_spec.get("inputs", []):
        param_dict = p.to_dict()
        if p.name in required_inputs:
            param_dict["label"] = mark_required(param_dict["label"])
        params[p.name] = param_dict
        input_names.append(p.name)

    # Process model_inputs
    for p in node_spec.get("model_inputs", []):
        param_dict = p.to_dict()
        if p.name in required_model_inputs:
            param_dict["label"] = mark_required(param_dict["label"])
        params[p.name] = param_dict
        model_input_names.append(p.name)

    # Process outputs: add a prefix to the output name if it already exists as an input
    for p in node_spec.get("outputs", []):
        if p.name in input_names:
            # rename to out_<name>
            output_name = f"out_{p.name}"
        else:
            output_name = p.name
        params[output_name] = p.to_dict()
        output_names.append(output_name)

    return {
        "params": params,
        "input_names": input_names,
        "model_input_names": model_input_names,
        "output_names": output_names,
        "block_name": node_spec.get("block_name"),
        "node_type": node_type,
    }


class MoDiffPipelineConfig:
    """
    Configuration for an entire MoDiff pipeline containing multiple nodes.

    Accepts node specs as dicts with inputs/model_inputs/outputs lists of MoDiffParam, converts them to MoDiff-ready
    format, and handles save/load to Hub.

    Example:
        ```python
        config = MoDiffPipelineConfig(
            node_specs={
                "denoise": {
                    "inputs": [MoDiffParam.seed(), MoDiffParam.prompt()],
                    "model_inputs": [MoDiffParam.unet()],
                    "outputs": [MoDiffParam.latents(display="output")],
                    "required_inputs": ["prompt"],
                    "required_model_inputs": ["unet"],
                    "block_name": "denoise",
                },
                "decoder": {
                    "inputs": [MoDiffParam.latents(display="input")],
                    "outputs": [MoDiffParam.images()],
                    "block_name": "decoder",
                },
            },
            label="My Pipeline",
            default_repo="user/my-pipeline",
            default_dtype="float16",
        )

        # Access MoDiff format dict
        denoise = config.node_params["denoise"]
        input_names = denoise["input_names"]
        params = denoise["params"]

        # Save to Hub
        config.save("./my_config", push_to_hub=True, repo_id="user/my-pipeline")

        # Load from Hub
        loaded = MoDiffPipelineConfig.load("user/my-pipeline")
        ```
    """

    config_name = "modiff_pipeline_config.json"

    def __init__(
        self,
        node_specs: dict[str, dict[str, Any] | None],
        label: str = "",
        default_repo: str = "",
        default_dtype: str = "",
        loader_component_outputs: tuple[str, ...] = (),
        layer_block_options: tuple[str, ...] = (),
        guider_options: tuple[str, ...] = (),
        scheduler_options: tuple[str, ...] = (),
        denoise_image_latent_dimensions: tuple[str, ...] = (),
    ):
        """
        Args:
            node_specs: Dict mapping node_type to node spec or None.
                        Node spec has: inputs, model_inputs, outputs, required_inputs, required_model_inputs,
                        block_name (all optional)
            label: Human-readable label for the pipeline
            default_repo: Default HuggingFace repo for this pipeline
            default_dtype: Default dtype (e.g., "float16", "bfloat16")
            loader_component_outputs: Additional required component names that ModelsLoader publishes.
            layer_block_options: Exact installed transformer block paths accepted by the Layers node.
            guider_options: Exact Diffusers guider classes accepted for this pipeline.
            scheduler_options: Exact Diffusers scheduler replacements accepted for this pipeline.
            denoise_image_latent_dimensions: Legacy dimension inputs retained when image latents are supplied.
        """
        # Convert all node specs to MoDiff format immediately
        self.node_specs = node_specs

        self.label = label
        self.default_repo = default_repo
        self.default_dtype = default_dtype
        if not isinstance(loader_component_outputs, (list, tuple)):
            raise ValueError("loader_component_outputs requires a list or tuple of component names.")
        normalized_loader_outputs = tuple(loader_component_outputs)
        if (
            len(normalized_loader_outputs) > MAX_LOADER_COMPONENT_OUTPUTS
            or any(
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 128
                or name in PROTOTYPE_SENSITIVE_FIELD_NAMES
                for name in normalized_loader_outputs
            )
            or len(normalized_loader_outputs) != len(set(normalized_loader_outputs))
        ):
            raise ValueError("loader_component_outputs requires bounded, unique component names.")
        self.loader_component_outputs = normalized_loader_outputs
        if not isinstance(layer_block_options, (list, tuple)):
            raise ValueError("layer_block_options requires a list or tuple of block names.")
        normalized_layer_blocks = tuple(layer_block_options)
        if (
            len(normalized_layer_blocks) > MAX_LAYER_BLOCK_OPTIONS
            or any(
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 256
                or name in PROTOTYPE_SENSITIVE_FIELD_NAMES
                for name in normalized_layer_blocks
            )
            or len(normalized_layer_blocks) != len(set(normalized_layer_blocks))
        ):
            raise ValueError("layer_block_options requires bounded, unique block names.")
        self.layer_block_options = normalized_layer_blocks
        if not isinstance(guider_options, (list, tuple)):
            raise ValueError("guider_options requires a list or tuple of guider class names.")
        normalized_guider_options = tuple(guider_options)
        if (
            len(normalized_guider_options) > MAX_GUIDER_OPTIONS
            or any(
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 128
                or name in PROTOTYPE_SENSITIVE_FIELD_NAMES
                for name in normalized_guider_options
            )
            or len(normalized_guider_options) != len(set(normalized_guider_options))
        ):
            raise ValueError("guider_options requires bounded, unique guider class names.")
        self.guider_options = normalized_guider_options
        if not isinstance(scheduler_options, (list, tuple)):
            raise ValueError("scheduler_options requires a list or tuple of scheduler class names.")
        normalized_scheduler_options = tuple(scheduler_options)
        if (
            len(normalized_scheduler_options) > MAX_SCHEDULER_OPTIONS
            or any(
                not isinstance(name, str)
                or not name.strip()
                or len(name) > 128
                or name in PROTOTYPE_SENSITIVE_FIELD_NAMES
                for name in normalized_scheduler_options
            )
            or len(normalized_scheduler_options) != len(set(normalized_scheduler_options))
        ):
            raise ValueError("scheduler_options requires bounded, unique scheduler class names.")
        self.scheduler_options = normalized_scheduler_options
        if not isinstance(denoise_image_latent_dimensions, (list, tuple)):
            raise ValueError("denoise_image_latent_dimensions requires a list or tuple of dimension names.")
        normalized_denoise_dimensions = tuple(denoise_image_latent_dimensions)
        if (
            len(normalized_denoise_dimensions) > MAX_DENOISE_IMAGE_LATENT_DIMENSIONS
            or any(
                not isinstance(name, str) or name not in SUPPORTED_DENOISE_IMAGE_LATENT_DIMENSIONS
                for name in normalized_denoise_dimensions
            )
            or len(normalized_denoise_dimensions) != len(set(normalized_denoise_dimensions))
        ):
            raise ValueError("denoise_image_latent_dimensions requires bounded, unique supported dimension names.")
        self.denoise_image_latent_dimensions = normalized_denoise_dimensions

    @property
    def node_params(self) -> dict[str, Any]:
        """Lazily compute node_params from node_specs."""
        if self.node_specs is None:
            return self._node_params

        params = {}
        for node_type, spec in self.node_specs.items():
            if spec is None:
                params[node_type] = None
            else:
                params[node_type] = node_spec_to_modiff_dict(spec, node_type)
        return params

    def __repr__(self) -> str:
        lines = [
            f"MoDiffPipelineConfig(label={self.label!r}, default_repo={self.default_repo!r}, default_dtype={self.default_dtype!r})"
        ]
        for node_type, spec in self.node_specs.items():
            if spec is None:
                lines.append(f"  {node_type}: None")
            else:
                inputs = [p.name for p in spec.get("inputs", [])]
                model_inputs = [p.name for p in spec.get("model_inputs", [])]
                outputs = [p.name for p in spec.get("outputs", [])]
                lines.append(f"  {node_type}:")
                lines.append(f"    inputs: {inputs}")
                lines.append(f"    model_inputs: {model_inputs}")
                lines.append(f"    outputs: {outputs}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "label": self.label,
            "default_repo": self.default_repo,
            "default_dtype": self.default_dtype,
            "loader_component_outputs": list(self.loader_component_outputs),
            "layer_block_options": list(self.layer_block_options),
            "guider_options": list(self.guider_options),
            "scheduler_options": list(self.scheduler_options),
            "denoise_image_latent_dimensions": list(self.denoise_image_latent_dimensions),
            "node_params": self.node_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoDiffPipelineConfig":
        """
        Create from a dictionary (loaded from JSON).

        Note: The modiff_params are already in MoDiff format when loading from JSON.
        """
        instance = cls.__new__(cls)
        instance.node_specs = None
        instance._node_params = data.get("node_params", {})
        instance.label = data.get("label", "")
        instance.default_repo = data.get("default_repo", "")
        instance.default_dtype = data.get("default_dtype", "")
        instance.loader_component_outputs = tuple(data.get("loader_component_outputs", ()))
        instance.layer_block_options = tuple(data.get("layer_block_options", ()))
        instance.guider_options = tuple(data.get("guider_options", ()))
        instance.scheduler_options = tuple(data.get("scheduler_options", ()))
        instance.denoise_image_latent_dimensions = tuple(data.get("denoise_image_latent_dimensions", ()))
        return instance

    def to_json_string(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike):
        """Save to a JSON file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @classmethod
    def from_json_bytes(cls, raw_bytes: bytes, *, source_label: str = "<memory>") -> "MoDiffPipelineConfig":
        """Load one bounded, duplicate-free JSON object from an exact byte sequence."""

        if len(raw_bytes) > MAX_MODIFF_PIPELINE_CONFIG_BYTES:
            raise EnvironmentError(
                f"The Modular Diffusers config at '{source_label}' is larger than the "
                f"{MAX_MODIFF_PIPELINE_CONFIG_BYTES}-byte limit."
            )
        data = _decode_pipeline_config_bytes(raw_bytes, source_label=source_label)
        _validate_pipeline_config_document(data, source_label=source_label)
        return cls.from_dict(data)

    @classmethod
    def from_json_file(cls, json_file_path: str | os.PathLike) -> "MoDiffPipelineConfig":
        """Load from a JSON file."""
        config_path = Path(json_file_path)
        raw_bytes = _read_pipeline_config_bytes(config_path)
        return cls.from_json_bytes(raw_bytes, source_label=str(config_path))

    def save(self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs):
        """Save the modiff pipeline config to a directory."""
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        output_path = os.path.join(save_directory, self.config_name)
        self.to_json_file(output_path)
        logger.info(f"Pipeline config saved to {output_path}")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

            upload_file(
                path_or_fileobj=output_path,
                path_in_repo=self.config_name,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message or "Upload MoDiffPipelineConfig",
                create_pr=create_pr,
            )
            logger.info(f"Pipeline config pushed to hub: {repo_id}")

    @classmethod
    def load_verified(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *,
        source: str,
        revision: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        token: bool | str | None = None,
    ) -> VerifiedMoDiffPipelineConfig:
        """Read a source-explicit sidecar and loader metadata without network access.

        ``source='hub'`` never falls back to a same-named local directory. It
        resolves only an exact cached Hub commit. ``source='local'`` never calls
        the Hub and rejects a sidecar symlink that escapes the selected model
        directory. The executable manifest detects reviewed config/Python drift;
        it is not an atomic code authorization or a model-weight proof.
        """

        if not isinstance(source, str) or source not in {"hub", "local"}:
            raise ValueError("Custom Modular Diffusers repositories must declare source as 'hub' or 'local'.")

        repository = str(pretrained_model_name_or_path or "").strip()
        if not repository:
            raise ValueError("Custom Modular Diffusers repositories require a non-empty repository or local path.")
        if len(repository) > MAX_CUSTOM_PIPELINE_REPOSITORY_CHARS:
            raise ValueError("Custom Modular Diffusers repository or local path exceeds 4096 characters.")

        normalized_revision = str(revision or "").strip() or None
        if source == "hub":
            try:
                validate_repo_id(repository)
            except ValueError as error:
                raise ValueError(f"Invalid Hugging Face repository ID {repository!r}: {error}") from error
            if normalized_revision is None or _IMMUTABLE_HUB_REVISION.fullmatch(normalized_revision) is None:
                raise ValueError(
                    "Custom Modular Diffusers Hub repositories require an immutable lowercase 40-character commit "
                    "revision before their MoDiff sidecar can be read."
                )
            try:
                config_file = hf_hub_download(
                    repository,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    local_files_only=True,
                    token=token,
                    revision=normalized_revision,
                )
            except (
                RepositoryNotFoundError,
                RevisionNotFoundError,
                EntryNotFoundError,
                HfHubHTTPError,
                ValueError,
            ) as error:
                raise EnvironmentError(
                    f"Could not resolve cached {cls.config_name} for {repository}@{normalized_revision}. "
                    "Install that exact revision through Model Manager before refreshing the custom pipeline contract."
                ) from error
            config_path = Path(config_file).absolute()
            if not config_path.is_file():
                raise EnvironmentError(
                    f"The cached Hub snapshot for {repository}@{normalized_revision} has no {cls.config_name}."
                )
            repository_path = config_path.parent
            _validate_hub_snapshot_config_path(config_path, revision=normalized_revision)
            normalized_repository = repository
            executable_manifest_sha256 = _executable_manifest_sha256(repository_path, source="hub")
        else:
            if normalized_revision is not None:
                raise ValueError(
                    "Local custom Modular Diffusers directories are mutable and must not claim a Hub revision. "
                    "Clear Revision or select the Hub source."
                )
            try:
                repository_path = Path(repository).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise EnvironmentError(
                    f"Local custom Modular Diffusers directory '{repository}' does not exist."
                ) from error
            if not repository_path.is_dir():
                raise EnvironmentError(f"Local custom Modular Diffusers source '{repository}' must be a directory.")
            try:
                config_path = (repository_path / cls.config_name).resolve(strict=True)
                config_path.relative_to(repository_path)
            except (OSError, RuntimeError, ValueError) as error:
                raise EnvironmentError(
                    f"Local {cls.config_name} must be a regular file contained by '{repository_path}'."
                ) from error
            if not config_path.is_file():
                raise EnvironmentError(f"No file named {cls.config_name} found in {repository_path}")
            normalized_repository = str(repository_path)
            executable_manifest_sha256 = _executable_manifest_sha256(repository_path, source="local")

        raw_bytes = _read_pipeline_config_bytes(config_path)
        config = cls.from_json_bytes(raw_bytes, source_label=str(config_path))
        return VerifiedMoDiffPipelineConfig(
            config=config,
            raw_bytes=raw_bytes,
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            source=source,
            repo_id=normalized_repository,
            revision=normalized_revision,
            executable_manifest_sha256=executable_manifest_sha256,
            config_path=str(config_path),
            repository_path=str(repository_path),
        )

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "MoDiffPipelineConfig":
        """Load a pipeline config from a local path or Hugging Face Hub."""
        cache_dir = kwargs.pop("cache_dir", None)
        local_dir = kwargs.pop("local_dir", None)
        local_dir_use_symlinks = kwargs.pop("local_dir_use_symlinks", "auto")
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            if not os.path.isfile(config_file):
                raise EnvironmentError(f"No file named {cls.config_name} found in {pretrained_model_name_or_path}")
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    local_dir=local_dir,
                    local_dir_use_symlinks=local_dir_use_symlinks,
                )
            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
                    " listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
                    " token having permission to this repo with `token` or log in with `hf auth login`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
                    " this model name. Check the model page at"
                    f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}."
                )
            except HfHubHTTPError as err:
                raise EnvironmentError(
                    "There was a specific connection error when trying to load"
                    f" {pretrained_model_name_or_path}:\n{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a {cls.config_name} file.\nCheckout your internet connection or see how to"
                    " run the library in offline mode at"
                    " 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {cls.config_name} file"
                )

        try:
            return cls.from_json_file(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"The config file at '{config_file}' is not a valid JSON file.")

    @classmethod
    def from_blocks(
        cls,
        blocks,
        template: dict[str, dict[str, Any]] | None = None,
        label: str = "",
        default_repo: str = "",
        default_dtype: str = "bfloat16",
    ) -> "MoDiffPipelineConfig":
        """
        Create MoDiffPipelineConfig by matching template against actual pipeline blocks.
        """
        if template is None:
            template = DEFAULT_NODE_SPECS

        sub_block_map = dict(blocks.sub_blocks)

        def filter_spec_for_block(template_spec: dict[str, Any], block) -> dict[str, Any] | None:
            """Filter template spec params based on what the block actually supports."""
            block_input_names = set(block.input_names)
            block_output_names = set(block.intermediate_output_names)
            block_component_names = set(block.component_names)

            filtered_inputs = [
                p
                for p in template_spec.get("inputs", [])
                if p.required_block_params is None
                or all(name in block_input_names for name in p.required_block_params)
            ]
            filtered_model_inputs = [
                p
                for p in template_spec.get("model_inputs", [])
                if p.required_block_params is None
                or all(name in block_component_names for name in p.required_block_params)
            ]
            filtered_outputs = [
                p
                for p in template_spec.get("outputs", [])
                if p.required_block_params is None
                or all(name in block_output_names for name in p.required_block_params)
            ]

            filtered_input_names = {p.name for p in filtered_inputs}
            filtered_model_input_names = {p.name for p in filtered_model_inputs}

            filtered_required_inputs = [
                r for r in template_spec.get("required_inputs", []) if r in filtered_input_names
            ]
            filtered_required_model_inputs = [
                r for r in template_spec.get("required_model_inputs", []) if r in filtered_model_input_names
            ]

            return {
                "inputs": filtered_inputs,
                "model_inputs": filtered_model_inputs,
                "outputs": filtered_outputs,
                "required_inputs": filtered_required_inputs,
                "required_model_inputs": filtered_required_model_inputs,
                "block_name": template_spec.get("block_name"),
            }

        # Build node specs
        node_specs = {}
        for node_type, template_spec in template.items():
            if template_spec is None:
                node_specs[node_type] = None
                continue

            block_name = template_spec.get("block_name")
            if block_name is None or block_name not in sub_block_map:
                node_specs[node_type] = None
                continue

            node_specs[node_type] = filter_spec_for_block(template_spec, sub_block_map[block_name])

        return cls(
            node_specs=node_specs,
            label=label or getattr(blocks, "model_name", ""),
            default_repo=default_repo,
            default_dtype=default_dtype,
        )

    @classmethod
    def from_custom_block(
        cls,
        block,
        node_label: str = None,
        input_types: dict[str, Any] | None = None,
        output_types: dict[str, Any] | None = None,
    ) -> "MoDiffPipelineConfig":
        """
        Create a MoDiffPipelineConfig from a custom block.

        Args:
            block: A block instance with `inputs`, `outputs`, and `expected_components`/`component_names` properties.
                Each InputParam/OutputParam should have metadata={"modiff": "<type>"} where type is one of: image,
                video, text, checkbox, number, slider, dropdown, model. If metadata is None, maps to "custom".
            node_label: The display label for the node. Defaults to block class name with spaces.
            input_types:
                Optional dict mapping input param names to modiff types. Overrides the block's metadata if provided.
                Example: {"prompt": "textbox", "image": "image"}
            output_types:
                Optional dict mapping output param names to modiff types. Overrides the block's metadata if provided.
                Example: {"prompt": "text", "images": "image"}

        Returns:
            MoDiffPipelineConfig instance
        """
        if node_label is None:
            class_name = block.__class__.__name__
            node_label = "".join([" " + c if c.isupper() else c for c in class_name]).strip()

        if input_types is None:
            input_types = {}
        if output_types is None:
            output_types = {}

        inputs = []
        model_inputs = []
        outputs = []
        required_inputs = []

        # Process block inputs
        for input_param in block.inputs:
            if input_param.name is None:
                continue
            if input_param.name in input_types:
                input_param = copy.copy(input_param)
                input_param.metadata = {"modiff": input_types[input_param.name]}
            if input_param.required:
                required_inputs.append(input_param.name)
            inputs.append(input_param_to_modiff_param(input_param))

        # Process block outputs
        for output_param in block.outputs:
            if output_param.name is None:
                continue
            if output_param.name in output_types:
                output_param = copy.copy(output_param)
                output_param.metadata = {"modiff": output_types[output_param.name]}
            outputs.append(output_param_to_modiff_param(output_param))

        # Process expected components (all map to model inputs)
        component_names = block.component_names
        for component_name in component_names:
            model_inputs.append(MoDiffParam.Input.model(component_name))

        # Always add doc output
        outputs.append(MoDiffParam.doc())

        node_spec = {
            "inputs": inputs,
            "model_inputs": model_inputs,
            "outputs": outputs,
            "required_inputs": required_inputs,
            "required_model_inputs": [],
            "block_name": "custom",
        }

        return cls(
            node_specs={"custom": node_spec},
            label=node_label,
        )
