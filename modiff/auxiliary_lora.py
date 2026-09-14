"""Immutable, model-neutral identity contract for Diffusers LoRA weights.

The graph-visible descriptor is intentionally self-contained, but never
trusted by a loader.  Every consumer validates the schema, resolves the exact
cached or local Safetensors alias, and hashes the bytes again immediately
before calling Diffusers. Header validation rejects malformed or empty files;
model/component key compatibility remains Diffusers' responsibility and is not
transactional across a multi-adapter load.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


LORA_DESCRIPTOR_SCHEMA = "modiff.diffusers-lora.v1"
MAX_LORA_ADAPTERS = 32
MAX_DESCRIPTOR_BYTES = 32 * 1024
MAX_SCHEDULER_CONFIG_BYTES = 16 * 1024
MAX_JSON_DEPTH = 8
MAX_JSON_VALUES = 512
MAX_JSON_CONTAINER_ITEMS = 256
MAX_JSON_STRING_CHARS = 4096

_EXACT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_EXACT_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEDULER_EXPORT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCHEDULER_CONFIG_CONTROL_KEYS = {
    "pretrained_model_name_or_path",
    "return_unused_kwargs",
}
_FLOW_MATCH_SCHEDULER = "FlowMatchEulerDiscreteScheduler"
_FLOW_MATCH_DEFAULTS = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": False,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "invert_sigmas": False,
    "shift_terminal": None,
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "time_shift_type": "exponential",
    "stochastic_sampling": False,
}
_FLOW_MATCH_BOOLEAN_FIELDS = {
    "use_dynamic_shifting",
    "invert_sigmas",
    "use_karras_sigmas",
    "use_exponential_sigmas",
    "use_beta_sigmas",
    "stochastic_sampling",
}
_FLOW_MATCH_SEQUENCE_FIELDS = {"base_image_seq_len", "max_image_seq_len"}
_FLOW_MATCH_OPTIONAL_SHIFT_FIELDS = {"base_shift", "max_shift"}
_FLOW_MATCH_MAX_TRAIN_TIMESTEPS = 100_000
_FLOW_MATCH_MAX_SEQUENCE_LENGTH = 1_000_000
_DESCRIPTOR_KEYS = {
    "schema",
    "artifact",
    "adapter_name",
    "scale",
    "scheduler",
    "descriptor_sha256",
}
_HUB_ARTIFACT_KEYS = {"source", "repository", "revision", "weight_name", "sha256"}
_LOCAL_ARTIFACT_KEYS = {"source", "root", "weight_name", "sha256"}
_SCHEDULER_KEYS = {"class_name", "config"}
_CONTROLLED_LORA_NODE_CONTRACTS = {
    ("modules.ModularDiffusers", "Lora"): ("model", None, 1.0),
    ("modules.DiffusersImage", "LoadAdapter"): ("adapter_path", "default", 1.0),
    ("modules.DiffusersAudio", "LoadAdapter"): ("adapter_path", "audio_style", 0.7),
}


@dataclass(frozen=True)
class ResolvedLoraDescriptor:
    """A descriptor whose exact weight bytes were revalidated for loading."""

    descriptor_sha256: str
    source: str
    repository: str | None
    revision: str | None
    load_directory: Path
    weight_name: str
    content_sha256: str
    adapter_name: str
    scale: float
    scheduler_class_name: str | None
    scheduler_config: dict[str, Any]


def _reject_duplicate_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate scheduler JSON key {key!r} is not allowed.")
        value[key] = item
    return value


def _reject_nonfinite_json_number(value):
    raise ValueError(f"Non-finite scheduler JSON number {value!r} is not allowed.")


def _validate_json_shape(value: Any, *, description: str) -> None:
    pending = [(value, 0)]
    value_count = 0
    while pending:
        item, depth = pending.pop()
        value_count += 1
        if value_count > MAX_JSON_VALUES:
            raise ValueError(f"{description} exceeds the {MAX_JSON_VALUES}-value limit.")
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"{description} exceeds the {MAX_JSON_DEPTH}-level nesting limit.")
        if type(item) is dict:
            if len(item) > MAX_JSON_CONTAINER_ITEMS:
                raise ValueError(
                    f"{description} exceeds the {MAX_JSON_CONTAINER_ITEMS}-item object limit."
                )
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{description} object keys must be strings.")
                if len(key) > MAX_JSON_STRING_CHARS:
                    raise ValueError(f"{description} contains an oversized object key.")
                pending.append((nested, depth + 1))
        elif type(item) is list:
            if len(item) > MAX_JSON_CONTAINER_ITEMS:
                raise ValueError(
                    f"{description} exceeds the {MAX_JSON_CONTAINER_ITEMS}-item array limit."
                )
            pending.extend((nested, depth + 1) for nested in item)
        elif isinstance(item, str):
            if len(item) > MAX_JSON_STRING_CHARS:
                raise ValueError(
                    f"{description} contains a string longer than {MAX_JSON_STRING_CHARS} characters."
                )
        elif item is None or type(item) in {bool, int}:
            continue
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{description} contains a non-finite number.")
        else:
            raise TypeError(f"{description} contains unsupported value type {type(item).__name__}.")


def _canonical_json(value: Any, *, description: str, maximum_bytes: int) -> str:
    _validate_json_shape(value, description=description)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must contain finite JSON values only.") from error
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{description} exceeds the {maximum_bytes}-byte limit.")
    return encoded


def _parse_scheduler_config(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        if len(value) > MAX_SCHEDULER_CONFIG_BYTES or len(value.encode("utf-8")) > MAX_SCHEDULER_CONFIG_BYTES:
            raise ValueError(
                f"LoRA scheduler config exceeds the {MAX_SCHEDULER_CONFIG_BYTES}-byte limit."
            )
        try:
            value = json.loads(
                value or "{}",
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_number,
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"LoRA scheduler config must be valid JSON: {error}") from error
        except RecursionError as error:
            raise ValueError(
                f"LoRA scheduler config exceeds the {MAX_JSON_DEPTH}-level nesting limit."
            ) from error
    if type(value) is not dict:
        raise TypeError("LoRA scheduler config must decode to a JSON object.")
    encoded = _canonical_json(
        value,
        description="LoRA scheduler config",
        maximum_bytes=MAX_SCHEDULER_CONFIG_BYTES,
    )
    return json.loads(encoded)


def reviewed_scheduler_class(name: str):
    """Return one installed, top-level Diffusers SchedulerMixin export."""

    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("LoRA scheduler class must be an exact nonblank Diffusers export name.")
    if not _SCHEDULER_EXPORT.fullmatch(name):
        raise ValueError("LoRA scheduler class must be a simple Diffusers export name.")
    if name != _FLOW_MATCH_SCHEDULER:
        raise ValueError(
            f"LoRA scheduler class {name!r} is not in MoDiff's reviewed scheduler contract."
        )

    import diffusers
    from diffusers import SchedulerMixin

    scheduler_class = getattr(diffusers, name, None)
    if (
        not isinstance(scheduler_class, type)
        or scheduler_class is SchedulerMixin
        or not issubclass(scheduler_class, SchedulerMixin)
    ):
        raise ValueError(f"LoRA scheduler class {name!r} is not an installed Diffusers SchedulerMixin export.")
    if not str(getattr(scheduler_class, "__module__", "")).startswith("diffusers."):
        raise ValueError(f"LoRA scheduler class {name!r} is not owned by the installed Diffusers package.")
    if not callable(getattr(scheduler_class, "from_config", None)):
        raise ValueError(f"LoRA scheduler class {name!r} does not expose the reviewed Diffusers config API.")
    return scheduler_class


def _validate_scheduler_overrides(scheduler_class: type, config: dict[str, Any]) -> None:
    try:
        parameters = inspect.signature(scheduler_class.__init__).parameters
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"LoRA scheduler class {scheduler_class.__name__!r} does not expose a reviewable constructor."
        ) from error
    explicit_parameters = {
        name
        for name, parameter in parameters.items()
        if name != "self"
        and not name.startswith("_")
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    allowed = set(_FLOW_MATCH_DEFAULTS) if scheduler_class.__name__ == _FLOW_MATCH_SCHEDULER else set()
    if not allowed.issubset(explicit_parameters):
        raise ValueError(
            f"LoRA scheduler class {scheduler_class.__name__!r} no longer matches its reviewed constructor."
        )
    unsupported = sorted(
        name
        for name in config
        if name in _SCHEDULER_CONFIG_CONTROL_KEYS or name not in allowed
    )
    if unsupported:
        raise ValueError(
            f"LoRA scheduler config contains unsupported constructor parameters: {', '.join(unsupported)}."
        )


def _bounded_scheduler_float(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if type(value) not in {int, float}:
        raise TypeError(f"LoRA scheduler field {field!r} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise ValueError(
            f"LoRA scheduler field {field!r} must be between {minimum} and {maximum}."
        )
    return normalized


def _normalize_flow_match_config(config: dict[str, Any], *, partial: bool) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field, value in config.items():
        if field in _FLOW_MATCH_BOOLEAN_FIELDS:
            if type(value) is not bool:
                raise TypeError(f"LoRA scheduler field {field!r} must be a boolean.")
            normalized[field] = value
        elif field == "num_train_timesteps":
            if type(value) is not int or not 1 <= value <= _FLOW_MATCH_MAX_TRAIN_TIMESTEPS:
                raise ValueError(
                    "LoRA scheduler field 'num_train_timesteps' must be an integer between "
                    f"1 and {_FLOW_MATCH_MAX_TRAIN_TIMESTEPS}."
                )
            normalized[field] = value
        elif field in _FLOW_MATCH_SEQUENCE_FIELDS:
            if type(value) is not int or not 1 <= value <= _FLOW_MATCH_MAX_SEQUENCE_LENGTH:
                raise ValueError(
                    f"LoRA scheduler field {field!r} must be an integer between 1 and "
                    f"{_FLOW_MATCH_MAX_SEQUENCE_LENGTH}."
                )
            normalized[field] = value
        elif field == "shift":
            normalized[field] = _bounded_scheduler_float(
                value,
                field=field,
                minimum=1e-6,
                maximum=100.0,
            )
        elif field in _FLOW_MATCH_OPTIONAL_SHIFT_FIELDS:
            normalized[field] = _bounded_scheduler_float(
                value,
                field=field,
                minimum=1e-6,
                maximum=100.0,
                allow_none=True,
            )
        elif field == "shift_terminal":
            normalized[field] = _bounded_scheduler_float(
                value,
                field=field,
                minimum=0.0,
                maximum=0.999999,
                allow_none=True,
            )
        elif field == "time_shift_type":
            if type(value) is not str or value not in {"exponential", "linear"}:
                raise ValueError("LoRA scheduler field 'time_shift_type' must be 'exponential' or 'linear'.")
            normalized[field] = value
        else:
            raise ValueError(f"LoRA scheduler field {field!r} is not reviewed.")

    if partial:
        return normalized
    if set(normalized) != set(_FLOW_MATCH_DEFAULTS):
        raise ValueError("The effective FlowMatch scheduler config is incomplete.")
    if normalized["base_image_seq_len"] >= normalized["max_image_seq_len"]:
        raise ValueError("FlowMatch base_image_seq_len must be smaller than max_image_seq_len.")
    if normalized["use_dynamic_shifting"] and (
        normalized["base_shift"] is None or normalized["max_shift"] is None
    ):
        raise ValueError("FlowMatch dynamic shifting requires finite base_shift and max_shift values.")
    if sum(
        int(normalized[field])
        for field in ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas")
    ) > 1:
        raise ValueError("FlowMatch permits only one alternate sigma schedule.")
    return normalized


def reviewed_scheduler_effective_config(
    scheduler_class: type,
    current_config: Any,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Build one bounded config without forwarding ConfigMixin control kwargs."""

    if scheduler_class.__name__ != _FLOW_MATCH_SCHEDULER:
        raise ValueError(f"Scheduler {scheduler_class.__name__!r} has no reviewed MoDiff config contract.")
    if not isinstance(current_config, Mapping):
        raise TypeError("The pipeline scheduler config must be a mapping.")
    _validate_scheduler_overrides(scheduler_class, overrides)
    effective = {
        field: overrides.get(field, current_config.get(field, default))
        for field, default in _FLOW_MATCH_DEFAULTS.items()
    }
    return _normalize_flow_match_config(effective, partial=False)


def normalize_scheduler_contract(class_name: Any, config: Any) -> dict[str, Any] | None:
    if class_name in (None, ""):
        normalized_config = _parse_scheduler_config(config if config is not None else {})
        if normalized_config:
            raise ValueError("LoRA scheduler config requires an explicit reviewed scheduler class.")
        return None
    if not isinstance(class_name, str):
        raise TypeError("LoRA scheduler class must be a string.")
    scheduler_class = reviewed_scheduler_class(class_name)
    normalized_config = _parse_scheduler_config(config if config is not None else {})
    _validate_scheduler_overrides(scheduler_class, normalized_config)
    normalized_config = _normalize_flow_match_config(normalized_config, partial=True)
    return {
        "class_name": class_name,
        "config": normalized_config,
    }


def _exact_sha256(value: Any, *, required: bool) -> str | None:
    if value in (None, "") and not required:
        return None
    if not isinstance(value, str) or value != value.strip() or not _EXACT_SHA256.fullmatch(value):
        raise ValueError("LoRA SHA-256 must contain exactly 64 lowercase hexadecimal digits.")
    return value


def _exact_weight_name(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("A LoRA weight requires an exact relative filename.")
    if len(value) > 1024 or "\\" in value or ":" in value or "\x00" in value or value.startswith("/"):
        raise ValueError("A LoRA weight_name must be a safe relative repository filename.")
    parts = value.split("/")
    if len(parts) > 64 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("A LoRA weight_name must stay inside its selected repository or directory.")
    if not PurePosixPath(value).name.endswith(".safetensors"):
        raise ValueError("A LoRA weight_name must use a literal lowercase .safetensors alias.")
    return value


def _exact_repository(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or value.count("/") != 1:
        raise ValueError("A Hub LoRA requires an exact namespace/repository ID.")
    from utils.huggingface import validate_hf_repo_id

    try:
        validate_hf_repo_id(value)
    except Exception as error:
        raise ValueError("A Hub LoRA requires a valid namespace/repository ID.") from error
    return value


def _exact_revision(value: Any) -> str:
    if not isinstance(value, str) or value != value.strip() or not _EXACT_COMMIT.fullmatch(value):
        raise ValueError("A Hub LoRA requires a lowercase 40-character commit revision.")
    return value


def _exact_adapter_name(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("A LoRA adapter name must be an exact nonblank string.")
    if len(value) > 256 or any(ord(character) < 32 for character in value):
        raise ValueError("A LoRA adapter name is oversized or contains control characters.")
    return value


def generated_lora_adapter_name(name_seed: str, node_id: str) -> str:
    """Stable generated identity that is legal as a PyTorch ModuleDict key."""
    original = f"{name_seed or 'lora'}_{node_id}"
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", original)
    if safe == original and len(safe) <= 256:
        return safe
    # Keep sanitized filenames and nested graph identities distinct; the
    # artifact's original filename remains unchanged in the descriptor.
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:16]
    return f"{safe[:239]}_{digest}"


def _exact_scale(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("A LoRA scale must be a finite number between -20 and 20.")
    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("A LoRA scale must be a finite number between -20 and 20.") from error
    if not math.isfinite(scale) or not -20 <= scale <= 20:
        raise ValueError("A LoRA scale must be a finite number between -20 and 20.")
    return scale


def _sha256_file(path: Path) -> str:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as error:
        raise FileNotFoundError(f"LoRA weight could not be read: {path}") from error
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field, None) != getattr(after, field, None) for field in identity_fields):
        raise ValueError("LoRA weight changed while its SHA-256 was being verified.")
    return digest.hexdigest()


def _preflight_safetensors_file(path: Path) -> None:
    """Validate the Safetensors header without materializing tensor bodies."""

    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(path, framework="np", device="cpu") as handle:
            keys = list(handle.keys())
    except (OSError, SafetensorError, ValueError) as error:
        raise ValueError("LoRA weight must be a valid Safetensors file.") from error
    if not keys:
        raise ValueError("LoRA Safetensors weight must contain at least one tensor.")


def _managed_hub_alias(repository: str, revision: str, weight_name: str) -> tuple[Path, Path]:
    from utils import huggingface as huggingface_utils

    cached = huggingface_utils.cached_file_path(repository, weight_name, revision=revision)
    if not cached:
        raise FileNotFoundError(
            f"LoRA {repository}@{revision}/{weight_name} is not installed. "
            "Install the pinned file through Model Manager first."
        )
    cached_path = Path(cached).expanduser()
    if not cached_path.is_absolute():
        raise ValueError("Installed Hub LoRA cache entries must use absolute managed-cache paths.")

    configured_root = huggingface_utils.CONFIG.hf["cache_dir"] or huggingface_utils.HUGGINGFACE_HUB_CACHE
    lexical_root = Path(os.path.abspath(Path(configured_root).expanduser()))
    lexical_alias = Path(os.path.abspath(cached_path))
    expected_lexical_alias = (
        lexical_root
        / f"models--{repository.replace('/', '--')}"
        / "snapshots"
        / revision
        / Path(*PurePosixPath(weight_name).parts)
    )
    if os.path.normcase(str(lexical_alias)) != os.path.normcase(str(expected_lexical_alias)):
        raise ValueError(
            "Installed Hub LoRA cache lookup returned an alias outside the exact repository snapshot path."
        )
    try:
        cache_root = lexical_root.resolve(strict=True)
        expected_repo_root = cache_root / f"models--{repository.replace('/', '--')}"
        repo_root = expected_repo_root.resolve(strict=True)
        if repo_root != expected_repo_root:
            raise ValueError("Managed Hub repository root cannot be a link to another cache location.")
        expected_snapshot_root = repo_root / "snapshots" / revision
        snapshot_root = expected_snapshot_root.resolve(strict=True)
        if snapshot_root != expected_snapshot_root:
            raise ValueError("Managed Hub snapshot root cannot be a link to another revision.")
        expected_alias = snapshot_root.joinpath(*PurePosixPath(weight_name).parts)
        alias = expected_alias.parent.resolve(strict=True) / expected_alias.name
        if alias.parent != expected_alias.parent:
            raise ValueError("Managed Hub snapshot subdirectories cannot redirect the adapter alias.")
        relative_alias = alias.relative_to(snapshot_root).as_posix()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "Installed Hub LoRA alias is not contained in the exact managed repository snapshot."
        ) from error
    if relative_alias != weight_name or not alias.is_file():
        raise ValueError(
            "Installed Hub LoRA cache lookup did not preserve the exact pinned .safetensors snapshot alias."
        )

    resolved_weight = huggingface_utils.resolve_managed_hf_cache_file(alias)
    try:
        resolved_weight.relative_to(repo_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Installed Hub LoRA resolves outside its managed repository cache.") from error
    return alias, resolved_weight


def _local_alias(root_value: Any, weight_name: str) -> tuple[Path, Path, Path]:
    if not isinstance(root_value, str) or not root_value or root_value != root_value.strip():
        raise ValueError("A local LoRA descriptor requires an exact absolute root path.")
    raw_root = Path(root_value).expanduser()
    if not raw_root.is_absolute():
        raise ValueError("A local LoRA descriptor root must be absolute.")
    try:
        root = raw_root.resolve(strict=True)
        if root != raw_root or not root.is_dir():
            raise ValueError("A local LoRA descriptor root must already be resolved to an existing directory.")
        requested = root.joinpath(*PurePosixPath(weight_name).parts)
        alias = requested.parent.resolve(strict=True) / requested.name
        if alias.relative_to(root).as_posix() != weight_name or not alias.is_file():
            raise ValueError("Local LoRA weight is not the declared file inside its resolved root.")
        resolved_weight = alias.resolve(strict=True)
        resolved_weight.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("A local LoRA descriptor root"):
            raise
        raise ValueError("Local LoRA weight must resolve to a contained existing file.") from error
    if not resolved_weight.is_file():
        raise FileNotFoundError("Local LoRA weight is not a file.")
    return root, alias, resolved_weight


def _local_selection_alias(value: Any, weight_name: Any) -> tuple[Path, Path, str]:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("A local LoRA requires an exact file or directory path.")
    selected = Path(value).expanduser()
    try:
        if selected.is_dir():
            root = selected.resolve(strict=True)
            normalized_weight = _exact_weight_name(weight_name)
            root, alias, _ = _local_alias(str(root), normalized_weight)
            return root, alias, normalized_weight
        if not selected.is_file():
            raise FileNotFoundError(f"Local LoRA path does not exist: {selected}")
        supplied_weight = None if weight_name in (None, "") else _exact_weight_name(weight_name)
        root = selected.parent.resolve(strict=True)
        alias = root / selected.name
        normalized_weight = _exact_weight_name(alias.name)
        if supplied_weight is not None and supplied_weight != normalized_weight:
            raise ValueError("Local LoRA file selection and weight_name refer to different files.")
        root, alias, _ = _local_alias(str(root), normalized_weight)
        return root, alias, normalized_weight
    except (OSError, RuntimeError) as error:
        raise FileNotFoundError(f"Local LoRA path does not exist: {selected}") from error


def _descriptor_digest(payload: dict[str, Any]) -> str:
    encoded = _canonical_json(
        payload,
        description="LoRA descriptor",
        maximum_bytes=MAX_DESCRIPTOR_BYTES,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_lora_descriptor(
    *,
    selection: Any,
    weight_name: Any,
    revision: Any,
    expected_sha256: Any,
    adapter_name: Any,
    scale: Any,
    scheduler_class: Any = "",
    scheduler_config: Any = None,
) -> dict[str, Any]:
    """Build a canonical descriptor after resolving and hashing its bytes."""

    if not isinstance(selection, dict) or "source" not in selection or "value" not in selection:
        raise TypeError("A LoRA selection must explicitly provide source and value fields.")
    source = selection.get("source")
    if source not in {"hub", "local"}:
        raise ValueError("A LoRA selection source must be exactly 'hub' or 'local'.")
    normalized_name = _exact_adapter_name(adapter_name)
    normalized_scale = _exact_scale(scale)
    scheduler = normalize_scheduler_contract(
        scheduler_class,
        scheduler_config if scheduler_config is not None else {},
    )

    if source == "hub":
        repository = _exact_repository(selection.get("value"))
        normalized_revision = _exact_revision(revision)
        normalized_weight = _exact_weight_name(weight_name)
        expected_digest = _exact_sha256(expected_sha256, required=True)
        alias, _ = _managed_hub_alias(repository, normalized_revision, normalized_weight)
        actual_digest = _sha256_file(alias)
        if actual_digest != expected_digest:
            raise ValueError(
                f"LoRA {repository}@{normalized_revision}/{normalized_weight} failed its pinned SHA-256 verification."
            )
        _preflight_safetensors_file(alias)
        artifact = {
            "source": "hub",
            "repository": repository,
            "revision": normalized_revision,
            "weight_name": normalized_weight,
            "sha256": actual_digest,
        }
    else:
        if revision not in (None, ""):
            raise ValueError("A local LoRA cannot carry a Hub revision.")
        expected_digest = _exact_sha256(expected_sha256, required=False)
        root, alias, normalized_weight = _local_selection_alias(selection.get("value"), weight_name)
        actual_digest = _sha256_file(alias)
        if expected_digest is not None and actual_digest != expected_digest:
            raise ValueError("Local LoRA failed its supplied SHA-256 verification.")
        _preflight_safetensors_file(alias)
        artifact = {
            "source": "local",
            "root": str(root),
            "weight_name": normalized_weight,
            "sha256": actual_digest,
        }

    payload = {
        "schema": LORA_DESCRIPTOR_SCHEMA,
        "artifact": artifact,
        "adapter_name": normalized_name,
        "scale": normalized_scale,
        "scheduler": scheduler,
    }
    return {**payload, "descriptor_sha256": _descriptor_digest(payload)}


def _graph_param_value(node: Mapping[str, Any], key: str, default: Any = None) -> Any:
    params = node.get("params")
    if not isinstance(params, Mapping):
        return default
    param = params.get(key)
    if not isinstance(param, Mapping):
        return default
    return param.get("value", param.get("default", default))


def lora_resource_requirement(node_id: str, node: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect pinned Modular LoRA bytes and shapes without loading tensors.

    Budget float32 copies for loading/conversion plus the largest dense update.
    Shape compatibility remains the existing Diffusers adapter boundary's job.
    """
    if (node.get("module"), node.get("action")) != ("modules.ModularDiffusers", "Lora"):
        raise ValueError("This adapter does not declare a workflow Auto memory envelope.")
    params = node.get("params", {})
    if any(isinstance(param, Mapping) and param.get("sourceId") for param in params.values()):
        raise ValueError("Auto needs the adapter's exact artifact and settings before model loading.")
    # Reuse all path, hash, scheduler and scalar checks from graph admission.
    receipt = controlled_lora_receipts_from_graph({"nodes": {node_id: node}, "paths": [[node_id]]})[0]
    selection = _graph_param_value(node, "model")
    artifact = receipt["artifact"]
    if artifact["source"] == "hub":
        alias, _ = _managed_hub_alias(artifact["repository"], artifact["revision"], artifact["weightName"])
    else:
        _, alias, _ = _local_selection_alias(selection.get("value"), _graph_param_value(node, "weight_name"))
    from safetensors import safe_open

    with safe_open(alias, framework="np", device="cpu") as handle:
        shapes = [handle.get_slice(key).get_shape() for key in handle.keys()]
    elements = sum(math.prod(shape) for shape in shapes)
    # A LoRA rank decomposition can produce a dense matrix larger than either
    # factor. The largest dimension squared bounds that temporary update.
    largest_dimension = max((max(shape, default=0) for shape in shapes), default=0)
    budget = elements * 4 * 3 + largest_dimension * largest_dimension * 4
    return {"systemRamBytes": budget, "vramBytes": budget, "artifact": artifact, "tensorCount": len(shapes)}


def controlled_lora_receipts_from_graph(graph: Any) -> list[dict[str, Any]]:
    """Validate executable controlled LoRA nodes and return exact safe receipts.

    The API graph is the authority here: caller-supplied runtime receipts are
    deliberately ignored.  Node IDs are used only to reproduce the Modular
    adapter name; local filesystem roots remain represented by the descriptor
    digest and are never copied into the public receipt.
    """

    if not isinstance(graph, Mapping):
        return []
    nodes = graph.get("nodes")
    paths = graph.get("paths")
    if not isinstance(nodes, Mapping) or not isinstance(paths, list):
        return []
    nodes_by_id = {str(node_id): node for node_id, node in nodes.items()}

    executable_ids: list[str] = []
    seen_ids: set[str] = set()
    for path in paths:
        if not isinstance(path, list):
            continue
        for raw_node_id in path:
            node_id = str(raw_node_id)
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                executable_ids.append(node_id)

    controlled_ids = [
        node_id
        for node_id in executable_ids
        if isinstance(nodes_by_id.get(node_id), Mapping)
        and (nodes_by_id[node_id].get("module"), nodes_by_id[node_id].get("action"))
        in _CONTROLLED_LORA_NODE_CONTRACTS
    ]
    if len(controlled_ids) > MAX_LORA_ADAPTERS:
        raise ValueError(f"At most {MAX_LORA_ADAPTERS} executable LoRA adapters are supported.")

    receipts: list[dict[str, Any]] = []
    for node_id in controlled_ids:
        node = nodes_by_id[node_id]
        module = node.get("module")
        action = node.get("action")
        selection_key, default_adapter_name, default_scale = _CONTROLLED_LORA_NODE_CONTRACTS[
            (module, action)
        ]
        selection = _graph_param_value(node, selection_key)
        image_selection_is_empty = module == "modules.DiffusersImage" and (
            selection is None
            or isinstance(selection, str)
            and not selection.strip()
            or isinstance(selection, Mapping)
            and isinstance(selection.get("value"), str)
            and not selection["value"].strip()
        )
        if image_selection_is_empty:
            # The direct-image adapter explicitly supports a no-op empty
            # selection. It contributes no execution artifact receipt.
            continue
        if isinstance(selection, str):
            selection = {
                "source": "local" if module == "modules.DiffusersAudio" else "hub",
                "value": selection,
            }

        weight_name = _graph_param_value(
            node,
            "weight_name",
            "adapter_model.safetensors" if module == "modules.DiffusersAudio" else None,
        )
        if module == "modules.ModularDiffusers":
            requested_weight = str(weight_name or "")
            name_seed = PurePosixPath(requested_weight.replace("\\", "/")).stem
            if not name_seed and isinstance(selection, Mapping):
                name_seed = PurePosixPath(
                    str(selection.get("value") or "").replace("\\", "/")
                ).stem
            adapter_name = generated_lora_adapter_name(name_seed, node_id)
        else:
            adapter_name = _graph_param_value(node, "adapter_name", default_adapter_name)

        descriptor = build_lora_descriptor(
            selection=selection,
            weight_name=weight_name,
            revision=_graph_param_value(node, "revision", ""),
            expected_sha256=_graph_param_value(node, "expected_sha256", ""),
            adapter_name=adapter_name,
            scale=_graph_param_value(node, "scale", default_scale),
            scheduler_class=(
                _graph_param_value(node, "scheduler_class", "")
                if module == "modules.ModularDiffusers"
                else ""
            ),
            scheduler_config=(
                _graph_param_value(node, "scheduler_config", "{}")
                if module == "modules.ModularDiffusers"
                else None
            ),
        )
        if module == "modules.DiffusersImage" and not -2 <= descriptor["scale"] <= 2:
            raise ValueError("Diffusers image adapter scale must be between -2 and 2.")
        if module == "modules.DiffusersAudio" and not 0 <= descriptor["scale"] <= 2:
            raise ValueError("Diffusers audio adapter scale must be between 0 and 2.")

        replace_existing = None
        if module != "modules.ModularDiffusers":
            replace_existing = _graph_param_value(node, "replace_existing", True)
            if type(replace_existing) is not bool:
                raise TypeError("Diffusers adapter replace_existing must be a boolean.")

        artifact = descriptor["artifact"]
        safe_artifact = {
            "source": artifact["source"],
            "weightName": artifact["weight_name"],
            "sha256": artifact["sha256"],
        }
        if artifact["source"] == "hub":
            safe_artifact.update(
                repository=artifact["repository"],
                revision=artifact["revision"],
            )
        receipts.append(
            {
                "schemaVersion": 1,
                "kind": "diffusers_lora",
                "module": module,
                "action": action,
                "artifact": safe_artifact,
                "adapterName": descriptor["adapter_name"],
                "scale": descriptor["scale"],
                "scheduler": descriptor["scheduler"],
                "replaceExisting": replace_existing,
                "descriptorSha256": descriptor["descriptor_sha256"],
            }
        )
    return receipts


def resolve_lora_descriptor(value: Any) -> ResolvedLoraDescriptor:
    """Strictly revalidate one descriptor and its exact current bytes."""

    if type(value) is not dict:
        raise TypeError("LoRA loaders require a versioned descriptor from the generic LoRA node.")
    _validate_json_shape(value, description="LoRA descriptor")
    if set(value) != _DESCRIPTOR_KEYS:
        raise ValueError("LoRA descriptor fields do not match the supported versioned contract.")
    if value.get("schema") != LORA_DESCRIPTOR_SCHEMA:
        raise ValueError(f"Unsupported LoRA descriptor schema {value.get('schema')!r}.")
    supplied_descriptor_digest = _exact_sha256(value.get("descriptor_sha256"), required=True)
    raw_payload = {key: value[key] for key in _DESCRIPTOR_KEYS if key != "descriptor_sha256"}
    if _descriptor_digest(raw_payload) != supplied_descriptor_digest:
        raise ValueError("LoRA descriptor identity digest does not match its fields.")
    payload = deepcopy(raw_payload)

    artifact = payload.get("artifact")
    if type(artifact) is not dict:
        raise TypeError("LoRA descriptor artifact must be an object.")
    source = artifact.get("source")
    adapter_name = _exact_adapter_name(payload.get("adapter_name"))
    scale = _exact_scale(payload.get("scale"))
    scheduler_value = payload.get("scheduler")
    if scheduler_value is None:
        scheduler = None
    else:
        if type(scheduler_value) is not dict or set(scheduler_value) != _SCHEDULER_KEYS:
            raise ValueError("LoRA scheduler fields do not match the supported contract.")
        scheduler = normalize_scheduler_contract(
            scheduler_value.get("class_name"),
            scheduler_value.get("config"),
        )

    if source == "hub":
        if set(artifact) != _HUB_ARTIFACT_KEYS:
            raise ValueError("Hub LoRA artifact fields do not match the supported contract.")
        repository = _exact_repository(artifact.get("repository"))
        revision = _exact_revision(artifact.get("revision"))
        normalized_weight = _exact_weight_name(artifact.get("weight_name"))
        expected_digest = _exact_sha256(artifact.get("sha256"), required=True)
        alias, _ = _managed_hub_alias(repository, revision, normalized_weight)
    elif source == "local":
        if set(artifact) != _LOCAL_ARTIFACT_KEYS:
            raise ValueError("Local LoRA artifact fields do not match the supported contract.")
        repository = None
        revision = None
        normalized_weight = _exact_weight_name(artifact.get("weight_name"))
        expected_digest = _exact_sha256(artifact.get("sha256"), required=True)
        _, alias, _ = _local_alias(artifact.get("root"), normalized_weight)
    else:
        raise ValueError("LoRA descriptor source must be exactly 'hub' or 'local'.")

    actual_digest = _sha256_file(alias)
    if actual_digest != expected_digest:
        raise ValueError("LoRA weight content no longer matches its descriptor SHA-256.")
    _preflight_safetensors_file(alias)
    return ResolvedLoraDescriptor(
        descriptor_sha256=supplied_descriptor_digest,
        source=source,
        repository=repository,
        revision=revision,
        load_directory=alias.parent,
        weight_name=alias.name,
        content_sha256=actual_digest,
        adapter_name=adapter_name,
        scale=scale,
        scheduler_class_name=(scheduler or {}).get("class_name"),
        scheduler_config=deepcopy((scheduler or {}).get("config") or {}),
    )


def resolve_lora_descriptors(value: Any) -> list[ResolvedLoraDescriptor]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError("At least one LoRA descriptor is required.")
    if len(values) > MAX_LORA_ADAPTERS:
        raise ValueError(f"At most {MAX_LORA_ADAPTERS} LoRA adapters may be loaded together.")
    resolved = [resolve_lora_descriptor(item) for item in values]
    names = [item.adapter_name for item in resolved]
    if len(set(names)) != len(names):
        raise ValueError("Connected LoRA descriptors must use unique adapter names.")
    return resolved


def scheduler_override_contract(
    resolved: list[ResolvedLoraDescriptor],
) -> tuple[type, dict[str, Any]] | None:
    overrides = [
        (item.scheduler_class_name, item.scheduler_config)
        for item in resolved
        if item.scheduler_class_name is not None
    ]
    if not overrides:
        return None
    canonical = [
        (class_name, _canonical_json(config, description="LoRA scheduler config", maximum_bytes=MAX_SCHEDULER_CONFIG_BYTES))
        for class_name, config in overrides
    ]
    if any(item != canonical[0] for item in canonical[1:]):
        raise ValueError("Connected LoRAs declare incompatible scheduler contracts.")
    class_name, _ = canonical[0]
    return reviewed_scheduler_class(class_name), deepcopy(overrides[0][1])
